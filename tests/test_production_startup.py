"""Start passenger_wsgi.py in a fresh process set up like LiteSpeed's lswsgi: ASCII output and no thread caps."""
import json
import os
import subprocess
import sys

import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STARTUP = r'''
import json, os, sys
import passenger_wsgi, cv2
# Characters Gemini replies can contain (e umlaut, em dash, curly quotes), built with chr() so this command line stays
# ASCII: under the C locale Linux can't even decode a non-ASCII command
print('Citro' + chr(0xEB) + 'n ' + chr(0x2014) + ' ' + chr(0x201C) + 'good' + chr(0x201D))
print(json.dumps({
    'services_loaded': passenger_wsgi.services_loaded,
    'load_error': passenger_wsgi.load_error_msg,
    'opencv_threads': cv2.getNumThreads(),
    'blas_threads': os.environ.get('OPENBLAS_NUM_THREADS'),
    'stdout_encoding': sys.stdout.encoding,
}))
'''

THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


@pytest.fixture(scope='module')
def started():
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    env.update(PYTHONIOENCODING='ascii', PYTHONUTF8='0', LC_ALL='C', LANG='C')
    proc = subprocess.run([sys.executable, '-c', STARTUP], cwd=APP_DIR, env=env, capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stderr.decode('utf-8', 'replace')
    return json.loads(proc.stdout.decode('utf-8').strip().splitlines()[-1])


def test_services_load(started):
    assert started['services_loaded'], started['load_error']


def test_log_output_survives_non_ascii_text(started):
    # lswsgi's streams are ASCII; passenger_wsgi.py switches them to UTF-8 so a log line can never abort a step
    assert started['stdout_encoding'].lower().replace('-', '') == 'utf8'


def test_thread_pools_are_capped(started):
    # The server has 96 cores and the account allows 100 processes + threads in total. Uncapped, OpenCV started a
    # thread per core and starved PHP, which returned 503s.
    assert started['opencv_threads'] == 1
    assert started['blas_threads'] == '1'
