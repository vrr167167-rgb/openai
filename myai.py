#!/usr/bin/env python3
"""
myai - one-file, offline, terminal-only coding assistant.

ONE COMMAND SETUP (needs internet once) - see README / the bootstrap line.
It downloads this script, builds a private environment, installs the engine,
downloads the model, creates the trigger command and tests everything.

AFTER THAT (fully offline), in any terminal:
    myai            start chatting
    myai doctor     check that everything is healthy
    myai uninstall  remove everything this script created

Everything lives in ~/.myai (nothing is installed system-wide).
Problems are explained in plain words; technical details go to ~/.myai/myai.log
"""

import contextlib
import errno
import os
import shutil
import ssl
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

if sys.version_info < (3, 8):
    sys.exit("myai needs Python 3.8 or newer. Please install a newer Python.")

# ----------------------------------------------------------------------------
# SETTINGS - change these if you like
# ----------------------------------------------------------------------------
TRIGGER = "myai"  # the word you type in any terminal to start the chat

# Main link first, backup link second (both are the same Qwen2.5-Coder 1.5B model)
MODEL_URLS = [
    "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/"
    "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
]

# Prebuilt CPU engine files (avoids needing a C++ compiler on most machines)
WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"

SYSTEM_PROMPT = (
    "You are a concise coding assistant. When asked for code, give complete, "
    "compilable, working examples with a short explanation."
)

CTX = 4096          # how much text the model can see at once (tokens)
MAX_REPLY = 1024    # longest single answer (tokens)
MIN_MODEL_BYTES = 300 * 1024 * 1024
MIN_FREE_BYTES = 2 * 1024 ** 3

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------
IS_WIN = os.name == "nt"
HOME = os.path.expanduser("~")
APP_DIR = os.environ.get("MYAI_HOME") or os.path.join(HOME, ".myai")
VENV_DIR = os.path.join(APP_DIR, "venv")
VENV_PY = os.path.join(
    VENV_DIR, "Scripts" if IS_WIN else "bin", "python.exe" if IS_WIN else "python"
)
MODEL_FILE = os.path.join(APP_DIR, "model.gguf")
SCRIPT_COPY = os.path.join(APP_DIR, "myai.py")
LOG_FILE = os.path.join(APP_DIR, "myai.log")
MANIFEST = os.path.join(APP_DIR, "launcher.path")
UNIX_BIN_DIR = os.path.join(HOME, ".local", "bin")
WIN_BIN_DIR = os.path.join(APP_DIR, "bin")
PATH_MARK = "# added by myai"


class LinkError(RuntimeError):
    """The download link itself is bad (as opposed to a network hiccup)."""


# ----------------------------------------------------------------------------
# SMALL HELPERS: printing, logging, friendly errors
# ----------------------------------------------------------------------------
def say(msg=""):
    try:
        print(msg, flush=True)
    except Exception:
        pass


def log(msg):
    try:
        os.makedirs(APP_DIR, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
            f.write("[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))
    except Exception:
        pass


def log_tail(n=12):
    """Last lines of the tools' own output (hides myai's internal log lines)."""
    try:
        with open(LOG_FILE, encoding="utf-8", errors="replace") as f:
            lines = [ln for ln in f.readlines() if ln.strip() and not ln.startswith("[20")]
        return "".join(lines[-n:])
    except Exception:
        return ""


def explain(e):
    """Turn any exception into one plain sentence."""
    if isinstance(e, PermissionError):
        return ("Permission denied for %s. Pick another folder by setting the "
                "MYAI_HOME environment variable." % (getattr(e, "filename", None) or "a file"))
    if isinstance(e, OSError) and getattr(e, "errno", None) == errno.ENOSPC:
        return "The disk is full. Free up about 2 GB and run the command again."
    if isinstance(e, MemoryError):
        return "Not enough memory (RAM) for the model. Close other programs and try again."
    if isinstance(e, (urllib.error.URLError, ConnectionError, TimeoutError)):
        return "No internet connection, or the server could not be reached."
    return "%s: %s" % (type(e).__name__, e)


def fail(msg):
    say("\nPROBLEM: " + msg)
    say("Technical details (if any): %s" % LOG_FILE)
    sys.exit(1)


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def run_logged(cmd):
    """Run a command, sending its output to the log file. Returns exit code."""
    log("RUN " + " ".join(cmd))
    try:
        os.makedirs(APP_DIR, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as lf:
            return subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    except OSError as e:
        log("RUN failed: %s" % e)
        return 1


# ----------------------------------------------------------------------------
# STEP 1: private Python environment
# ----------------------------------------------------------------------------
def find_engine_python():
    """Prebuilt engine files exist for Python 3.10-3.12, so prefer those."""
    cur = sys.version_info[:2]
    if (3, 10) <= cur <= (3, 12) and sys.executable:
        return [sys.executable]
    for v in ("3.12", "3.11", "3.10"):
        exe = shutil.which("python" + v)
        if exe:
            return [exe]
    if IS_WIN and shutil.which("py"):
        for v in ("3.12", "3.11", "3.10"):
            try:
                r = subprocess.run(["py", "-" + v, "-c", "import sys"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=30)
                if r.returncode == 0:
                    return ["py", "-" + v]
            except (OSError, subprocess.TimeoutExpired):
                pass
    if not sys.executable:
        fail("Could not locate Python. Please install Python 3.12 and try again.")
    return [sys.executable]


def check_disk():
    try:
        if os.path.exists(MODEL_FILE):
            return
        free = shutil.disk_usage(APP_DIR).free
        if free < MIN_FREE_BYTES:
            fail("Not enough disk space: about 2 GB is needed, %.1f GB is free." % (free / 1e9))
    except OSError:
        pass  # cannot measure - carry on


def venv_healthy():
    if not os.path.exists(VENV_PY):
        return False
    try:
        r = subprocess.run([VENV_PY, "-m", "pip", "--version"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def make_venv():
    if venv_healthy():
        return
    say("[1/5] Creating private Python environment...")
    if os.path.isdir(VENV_DIR):
        shutil.rmtree(VENV_DIR, ignore_errors=True)  # broken/old one: rebuild
    py = find_engine_python()
    if sys.version_info[:2] >= (3, 13) and py == [sys.executable]:
        say("  Note: Python %d.%d has no prebuilt engine yet, so setup may need a C++"
            % sys.version_info[:2])
        say("  compiler. Installing Python 3.12 makes this much faster and easier.")
    run_logged(py + ["-m", "venv", VENV_DIR])
    if not venv_healthy():
        fail("Could not create the private Python environment.\n"
             "  Debian/Ubuntu: sudo apt install python3-venv python3-pip\n"
             "  Then run the same command again.\n" + log_tail(6))


# ----------------------------------------------------------------------------
# STEP 2: engine
# ----------------------------------------------------------------------------
def engine_ok():
    try:
        r = subprocess.run([VENV_PY, "-c", "import llama_cpp"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def install_engine():
    if engine_ok():
        return
    say("[2/5] Installing the model engine (llama-cpp-python)...")
    say("      This can take a minute. Please wait.")
    base = [
        VENV_PY, "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
        "--retries", "3", "--timeout", "30", "--prefer-binary",
        "llama-cpp-python", "--extra-index-url", WHEEL_INDEX,
    ]
    code = run_logged(base + ["--only-binary=llama-cpp-python"])
    if code != 0:
        say("      No prebuilt engine for this system. Building from source instead")
        say("      (can take 5-15 minutes and needs a C++ compiler)...")
        code = run_logged(base)
    if code != 0 or not engine_ok():
        fail("The engine could not be installed. Last lines of the log:\n" + log_tail(10) +
             "\nFixes: check your internet; install Python 3.12; or install a C++ compiler:\n"
             "  Mac:     xcode-select --install\n"
             "  Linux:   sudo apt install build-essential cmake\n"
             "  Windows: Visual Studio Build Tools (C++ workload)")


# ----------------------------------------------------------------------------
# STEP 3: model download (progress bar, resume, retries, backup link)
# ----------------------------------------------------------------------------
def model_ok(path):
    try:
        if os.path.getsize(path) < MIN_MODEL_BYTES:
            return False
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def fetch(url, part):
    done = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "myai/1.0"})
    if done:
        req.add_header("Range", "bytes=%d-" % done)
    resp = urllib.request.urlopen(req, timeout=30)
    if done and resp.getcode() != 206:
        done = 0  # server ignored the resume request; start again
    length = resp.headers.get("Content-Length")
    total = (int(length) + done) if length else 0
    with resp, open(part, "ab" if done else "wb") as f:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total:
                print("\r  %s / %s MB  (%d%%)" % (
                    format(done // 1000000, ","), format(total // 1000000, ","),
                    done * 100 // total), end="", flush=True)
            else:
                print("\r  %s MB" % format(done // 1000000, ","), end="", flush=True)
    print()
    if total and done < total:
        raise RuntimeError("connection dropped")


def curl_fetch(url, part):
    curl = shutil.which("curl")
    if not curl:
        return False
    say("  (using curl for the download)")
    return subprocess.call([curl, "-L", "--fail", "-C", "-", "-o", part, url]) == 0


def download(url, dest):
    """Returns normally on success; LinkError if the link is bad; RuntimeError otherwise."""
    part = dest + ".part"
    last_err = None
    for attempt in range(1, 6):
        try:
            fetch(url, part)
            break
        except urllib.error.HTTPError as e:
            if e.code == 416:  # nothing left to fetch
                if model_ok(part):
                    break
                _rm(part)
                last_err = e
            elif e.code in (429, 500, 502, 503, 504):
                last_err = e
            else:
                raise LinkError("The model link returned HTTP %d." % e.code)
        except urllib.error.URLError as e:
            if isinstance(e.reason, ssl.SSLError):
                if curl_fetch(url, part):
                    break
                raise RuntimeError(
                    "Secure connection failed (missing certificates). On a Mac with "
                    "python.org Python, run 'Install Certificates.command' from the "
                    "Python folder in Applications, then run the command again.")
            last_err = e
        except (OSError, RuntimeError) as e:
            last_err = e
        log("download attempt %d failed: %r" % (attempt, last_err))
        say("  Connection problem (%s). Retrying %d/5..." % (explain(last_err), attempt))
        time.sleep(min(3 * attempt, 15))
    else:
        raise RuntimeError("Download failed after several tries (%s)." % explain(last_err))
    if not model_ok(part):
        _rm(part)
        raise LinkError("The downloaded file is not a valid model.")
    os.replace(part, dest)


def get_model():
    if os.path.exists(MODEL_FILE):
        if model_ok(MODEL_FILE):
            return
        _rm(MODEL_FILE)
    say("[3/5] Downloading the model (about 1 GB, one time only)...")
    urls = ([os.environ["MYAI_MODEL_URL"]] if os.environ.get("MYAI_MODEL_URL") else []) + MODEL_URLS
    notes = []
    for i, url in enumerate(urls):
        if i:
            say("  Trying the backup download link...")
            _rm(MODEL_FILE + ".part")  # a partial file from another link would be corrupt
        try:
            download(url, MODEL_FILE)
            return
        except LinkError as e:
            log("link failed: %s (%s)" % (url, e))
            notes.append(str(e))
        except RuntimeError as e:
            fail("%s\nYour partial download is kept - run the same command again "
                 "and it will resume." % e)
    fail("Every download link failed: " + " ".join(notes))


# ----------------------------------------------------------------------------
# STEP 4: the trigger command
# ----------------------------------------------------------------------------
def _append_once(path, line):
    text = ""
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="surrogateescape") as f:
            text = f.read()
    if line in text:
        return
    with open(path, "a", encoding="utf-8", errors="surrogateescape") as f:
        if text and not text.endswith("\n"):
            f.write("\n")
        f.write("%s\n%s\n" % (PATH_MARK, line))


def _strip_path_lines(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        lines = f.read().split("\n")
    out, skip = [], False
    for ln in lines:
        if skip:
            skip = False
            continue
        if ln.strip() == PATH_MARK:
            skip = True
            continue
        out.append(ln)
    with open(path, "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write("\n".join(out))


def _rc_files():
    return [os.path.join(HOME, n) for n in (".bashrc", ".zshrc", ".profile")] + [
        os.path.join(HOME, ".config", "fish", "config.fish")]


def _on_path(d):
    want = os.path.normpath(d)
    return any(p and os.path.normpath(p) == want
               for p in os.environ.get("PATH", "").split(os.pathsep))


def add_to_path_unix():
    """Returns True if a shell start-up file was changed (new terminal needed)."""
    shell = os.environ.get("SHELL", "")
    export = 'export PATH="%s:$PATH"' % UNIX_BIN_DIR
    rcs = _rc_files()
    targets = [p for p in rcs[:3] if os.path.exists(p)]
    if "zsh" in shell and rcs[1] not in targets:
        targets.append(rcs[1])
    if not targets:
        targets = [rcs[2]]
    for rc in targets:
        _append_once(rc, export)
    if "fish" in shell:
        os.makedirs(os.path.dirname(rcs[3]), exist_ok=True)
        _append_once(rcs[3], 'set -gx PATH "%s" $PATH' % UNIX_BIN_DIR)
    return True


def _broadcast_env_change():
    try:
        import ctypes
        ctypes.windll.user32.SendMessageTimeoutW(
            0xFFFF, 0x1A, 0, "Environment", 0x2, 5000, None)
    except Exception:
        pass


def add_to_path_windows():
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                        winreg.KEY_READ | winreg.KEY_WRITE) as key:
        try:
            current, _ = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            current = ""
        if WIN_BIN_DIR.lower() in [p.lower() for p in current.split(";")]:
            return False
        new = (current.rstrip(";") + ";" if current else "") + WIN_BIN_DIR
        winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new)
    _broadcast_env_change()
    return True


def remove_from_path_windows():
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                        winreg.KEY_READ | winreg.KEY_WRITE) as key:
        try:
            current, _ = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            return
        parts = [p for p in current.split(";") if p and p.lower() != WIN_BIN_DIR.lower()]
        winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, ";".join(parts))
    _broadcast_env_change()


def fallback_command():
    return '"%s" "%s" chat' % (VENV_PY, SCRIPT_COPY)


def make_trigger():
    """Returns (launcher_path or None, new_terminal_needed)."""
    say("[4/5] Creating the '%s' command..." % TRIGGER)
    try:
        if IS_WIN:
            bin_dir, already_on_path = WIN_BIN_DIR, False
            launcher = os.path.join(bin_dir, TRIGGER + ".cmd")
        else:
            bin_dir, already_on_path = UNIX_BIN_DIR, False
            for d in (UNIX_BIN_DIR, os.path.join(HOME, "bin")):
                if _on_path(d):  # already on PATH: works instantly, no restart
                    bin_dir, already_on_path = d, True
                    break
            launcher = os.path.join(bin_dir, TRIGGER)
        os.makedirs(bin_dir, exist_ok=True)

        other = shutil.which(TRIGGER)
        if other and os.path.realpath(other) != os.path.realpath(launcher):
            say("  Warning: another command named '%s' already exists (%s)." % (TRIGGER, other))
            say("  Change TRIGGER at the top of myai.py to a different word if it clashes.")

        if IS_WIN:
            # no parenthesis blocks: a folder name like "John (Work)" would break them
            with open(launcher, "w") as f:
                f.write('@echo off\r\n'
                        'set "PY=%s"\r\n'
                        'if not exist "%%PY%%" set "PY=python"\r\n'
                        'if "%%~1"=="" goto chat\r\n'
                        '"%%PY%%" "%s" %%*\r\n'
                        'exit /b %%errorlevel%%\r\n'
                        ':chat\r\n'
                        '"%%PY%%" "%s" chat\r\n'
                        'exit /b %%errorlevel%%\r\n' % (VENV_PY, SCRIPT_COPY, SCRIPT_COPY))
            new_terminal = add_to_path_windows()
        else:
            # if the private environment is ever broken, fall back to system Python so
            # that 'myai doctor' / 'myai uninstall' still work and explain what to do
            with open(launcher, "w") as f:
                f.write('#!/bin/sh\n'
                        'PY="%s"\n'
                        'if [ ! -x "$PY" ]; then PY="$(command -v python3 || command -v python)"; fi\n'
                        'if [ -z "$PY" ]; then echo "myai: Python not found. Run the setup '
                        'command again."; exit 1; fi\n'
                        'if [ $# -eq 0 ]; then set -- chat; fi\n'
                        'exec "$PY" "%s" "$@"\n' % (VENV_PY, SCRIPT_COPY))
            os.chmod(launcher, 0o755)
            new_terminal = False if already_on_path else add_to_path_unix()
        with open(MANIFEST, "w", encoding="utf-8") as f:
            f.write(launcher)
        return launcher, new_terminal
    except Exception as e:  # never let the trigger step kill the setup
        log("make_trigger failed: %s" % traceback.format_exc())
        say("  Could not create the command (%s)." % explain(e))
        return None, False


# ----------------------------------------------------------------------------
# INSTALL (the whole one-step setup)
# ----------------------------------------------------------------------------
def install():
    say("\n=== Setting up '%s' (about 1 GB download, keep this window open) ===\n" % TRIGGER)
    try:
        if os.path.getsize(LOG_FILE) > 1_000_000:
            _rm(LOG_FILE)
    except OSError:
        pass
    try:
        me = os.path.realpath(os.path.abspath(__file__))
    except NameError:
        fail("Save myai.py to a file first, then run it with Python.")
    os.makedirs(APP_DIR, exist_ok=True)
    log("install started, python %s" % sys.version.split()[0])
    if me != os.path.realpath(SCRIPT_COPY):
        shutil.copyfile(me, SCRIPT_COPY)

    check_disk()
    make_venv()
    install_engine()
    get_model()
    launcher, new_terminal = make_trigger()

    say("[5/5] Testing the model (a few seconds)...")
    ok = subprocess.call([VENV_PY, SCRIPT_COPY, "selftest"]) == 0
    log("selftest ok=%s" % ok)

    say("\n=====================================================")
    if ok:
        say(" Download is completed.")
        say(" Pipeline setup is completed, ready for chatting.")
    else:
        say(" Downloaded and installed, but the model test FAILED (see above).")
        say(" If you saw 'illegal instruction', this CPU is too old for the")
        say(" prebuilt engine: install a C++ compiler and run the command again.")
    if launcher:
        if new_terminal:
            say(" Open a NEW terminal (in VS Code: close and reopen the terminal)")
            say(" and type:  %s" % TRIGGER)
        else:
            say(" Type:  %s" % TRIGGER)
    say(" If '%s' is not found, run this instead:" % TRIGGER)
    say("   %s" % fallback_command())
    say(" Check health any time with:  %s doctor" % TRIGGER)
    say(" From now on it runs fully offline.")
    say("=====================================================\n")


# ----------------------------------------------------------------------------
# CHAT
# ----------------------------------------------------------------------------
def setup_console():
    for s in (sys.stdout, sys.stderr, sys.stdin):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def ensure_in_venv(cmd):
    if os.path.realpath(sys.prefix) == os.path.realpath(VENV_DIR):
        return
    if not os.path.exists(VENV_PY):
        fail("Not set up yet. Run the setup command again.")
    sys.exit(subprocess.call([VENV_PY, os.path.abspath(__file__), cmd]))


@contextlib.contextmanager
def quiet_stderr():
    """Hide the engine's noisy start-up messages."""
    saved, fd = None, None
    try:
        fd = sys.stderr.fileno()
        saved = os.dup(fd)
        dn = os.open(os.devnull, os.O_WRONLY)
        os.dup2(dn, fd)
        os.close(dn)
    except Exception:
        saved = None
    try:
        yield
    finally:
        if saved is not None:
            os.dup2(saved, fd)
            os.close(saved)


def load_model():
    from llama_cpp import Llama
    with quiet_stderr():
        return Llama(model_path=MODEL_FILE, n_ctx=CTX, verbose=False)


def count_tokens(llm, msgs):
    return sum(len(llm.tokenize(m["content"].encode("utf-8"), add_bos=False)) + 8
               for m in msgs)


def trim_history(llm, history):
    """Drop the oldest exchanges until the conversation fits the context."""
    budget = CTX - MAX_REPLY - 64
    try:
        while len(history) > 2 and count_tokens(llm, history) > budget:
            del history[1:3]
    except Exception:
        del history[1:-1]  # cannot measure: keep only the newest message


def read_input():
    s = input("\nyou > ").strip()
    if s.startswith('"""'):  # multi-line paste mode: """ ... """
        if s.endswith('"""') and len(s) >= 6:
            return s[3:-3].strip()
        lines = [s[3:]]
        while True:
            line = input("...   ")
            if line.strip().endswith('"""'):
                lines.append(line.rstrip()[:-3])
                break
            lines.append(line)
        return "\n".join(lines).strip()
    return s


def selftest():
    ensure_in_venv("selftest")
    try:
        llm = load_model()
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            max_tokens=8, temperature=0)
        answer = out["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log("selftest failed: %s" % traceback.format_exc())
        fail("The model could not run: %s" % explain(e))
    say("  Model answered: %s" % answer)


def chat():
    ensure_in_venv("chat")
    if not model_ok(MODEL_FILE):
        fail("The model file is missing or damaged. Run the setup command again.")
    try:
        import readline  # noqa: F401  (arrow keys / history where available)
    except ImportError:
        pass

    say("Loading model, please wait...")
    try:
        llm = load_model()
    except Exception as e:
        log("load failed: %s" % traceback.format_exc())
        fail("Could not load the model: %s\nRun '%s doctor' to check the setup."
             % (explain(e), TRIGGER))

    say("\n" + "=" * 56)
    say(" %s - offline coding assistant (Qwen2.5-Coder 1.5B)" % TRIGGER)
    say(" Type a question and press Enter.")
    say(' Paste code:  start with """ and end with """')
    say(" /clear = new chat   /exit = quit   Ctrl+C = stop answer / quit")
    say("=" * 56)

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    errors_in_row = 0
    while True:
        try:
            q = read_input()
        except (EOFError, KeyboardInterrupt):
            say()
            break
        if not q:
            continue
        if q.lower() in ("/exit", "/quit", "exit", "quit"):
            break
        if q == "/clear":
            history = history[:1]
            say("Conversation cleared.")
            continue

        history.append({"role": "user", "content": q})
        trim_history(llm, history)

        reply = ""
        print("\nai  > ", end="", flush=True)
        try:
            for chunk in llm.create_chat_completion(
                    messages=history, max_tokens=MAX_REPLY,
                    temperature=0.2, stream=True):
                tok = chunk["choices"][0]["delta"].get("content")
                if tok:
                    print(tok, end="", flush=True)
                    reply += tok
            errors_in_row = 0
        except KeyboardInterrupt:
            print("\n[stopped]", end="")
        except ValueError:
            history.pop()
            print("[that message is too long for the model - please shorten it]")
            continue
        except Exception as e:  # never crash the chat: explain and carry on
            log("generation failed: %s" % traceback.format_exc())
            print("\n[The model hit a problem: %s]" % explain(e))
            history = history[:1]
            errors_in_row += 1
            if errors_in_row >= 3:
                say("Repeated problems. Close this and run '%s doctor'." % TRIGGER)
                break
            continue
        print()
        if reply:
            history.append({"role": "assistant", "content": reply})
        else:
            history.pop()


# ----------------------------------------------------------------------------
# DOCTOR / UNINSTALL
# ----------------------------------------------------------------------------
def _read_manifest():
    try:
        with open(MANIFEST, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def doctor():
    launcher = _read_manifest()
    healthy = venv_healthy()
    say("myai health check")
    say("  Install folder      : %s" % APP_DIR)
    say("  Python running this : %s" % sys.version.split()[0])
    say("  Private environment : %s" % ("OK" if healthy else "MISSING or BROKEN"))
    say("  Engine              : %s" % ("OK" if healthy and engine_ok() else "NOT working"))
    if model_ok(MODEL_FILE):
        say("  Model file          : OK (%d MB)" % (os.path.getsize(MODEL_FILE) // 1000000))
    else:
        say("  Model file          : MISSING or damaged")
    say("  Trigger file        : %s" % (launcher if launcher and os.path.exists(launcher)
                                        else "MISSING"))
    say("  Trigger on PATH     : %s" % ("yes" if shutil.which(TRIGGER)
                                        else "not in this terminal (open a new one)"))
    say("  Log file            : %s" % LOG_FILE)
    say("If anything says MISSING/BROKEN/NOT working, run the setup command again.")


def uninstall():
    launcher = _read_manifest()
    for p in {launcher, os.path.join(UNIX_BIN_DIR, TRIGGER),
              os.path.join(HOME, "bin", TRIGGER), os.path.join(WIN_BIN_DIR, TRIGGER + ".cmd")}:
        if p:
            _rm(p)
    if IS_WIN:
        try:
            remove_from_path_windows()
        except Exception:
            pass
    else:
        for rc in _rc_files():
            try:
                _strip_path_lines(rc)
            except OSError:
                pass
    shutil.rmtree(APP_DIR, ignore_errors=True)
    if os.path.exists(APP_DIR):
        say("Removed most of it. Delete this folder by hand to finish: %s" % APP_DIR)
    else:
        say("Removed everything myai created.")


# ----------------------------------------------------------------------------
# MAIN (with the crash guard)
# ----------------------------------------------------------------------------
def main():
    setup_console()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "install"
    try:
        if cmd in ("install", "setup"):
            install()
        elif cmd == "chat":
            chat()
        elif cmd == "selftest":
            selftest()
        elif cmd == "doctor":
            doctor()
        elif cmd == "uninstall":
            uninstall()
        else:
            say("Usage: %s [chat | doctor | uninstall]" % TRIGGER)
            sys.exit(2)
    except KeyboardInterrupt:
        say("\nStopped. Run the same command again to continue where you left off.")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        log("CRASH: %s" % traceback.format_exc())
        say("\nSomething went wrong: %s" % explain(e))
        say("Nothing is damaged - run the same command again and it will continue.")
        say("Technical details were saved to: %s" % LOG_FILE)
        sys.exit(1)


if __name__ == "__main__":
    main()
