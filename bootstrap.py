"""Create an isolated environment, initialize data/models, and launch Streamlit."""
from pathlib import Path
import argparse
import hashlib
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent

def run(command):
    subprocess.run([str(c) for c in command],cwd=ROOT,check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup-only', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--port', type=int,default=8501)
    args = parser.parse_args()
    if sys.version_info < (3,10):
        raise SystemExit('Please install Python 3.10 or newer.')
    env = ROOT/'.venv'
    python = env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    if not python.exists():
        print('Creating project-local Python environment...',flush=True)
        venv.EnvBuilder(with_pip=True).create(env)
    requirements = ROOT/'requirements.txt'
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = env/'.requirements.sha256'
    if not marker.exists() or marker.read_text().strip()!=digest:
        print('Installing project dependencies...',flush=True)
        run([python,'-m','pip','install','-r',requirements])
        marker.write_text(digest)
    run([python,ROOT/'engine.py','--setup'])
    if args.verify:
        run([python,'-m','unittest','discover','-s','tests','-v'])
    elif not args.setup_only:
        print(f'Open http://localhost:{args.port} in your browser. Press Ctrl+C to stop.',flush=True)
        run([python,'-m','streamlit','run',ROOT/'app.py','--server.address=127.0.0.1',
             f'--server.port={args.port}','--server.headless=true'])

if __name__=='__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f'Command failed (exit {exc.returncode}). See the error above.')
    except KeyboardInterrupt:
        print('\nInvoice Intelligence stopped.')
