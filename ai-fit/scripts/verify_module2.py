"""Run project regression checks using the active Python interpreter."""
import argparse,ast,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--node',default='node');args=parser.parse_args()
commands=[[sys.executable,'-B','test_app.py'],[sys.executable,'-B','test_sizing.py'],[sys.executable,'-B','-m','unittest','discover','-s','gpu_service','-p','test_*.py'],[args.node,'--check','static/app.js'],[args.node,'test_catalog.cjs'],[args.node,'test_generate.cjs'],[args.node,'test_module2_ui.cjs']]
results=[]
for command in commands:
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
    results.append({'command':command,'returncode':result.returncode,'output':result.stdout+result.stderr})
    print('PASS' if result.returncode==0 else 'FAIL',' '.join(command[2:] if command[0]==sys.executable else command[1:]),flush=True)
    if result.returncode: print(result.stdout+result.stderr,flush=True)
for p in [*ROOT.glob('*.py'),* (ROOT/'gpu_service').glob('*.py'),* (ROOT/'scripts').glob('*.py')]:ast.parse(p.read_text(encoding='utf-8-sig'))
(ROOT/'reports/module2-test-results.json').write_text(json.dumps({'python':sys.version,'results':results,'syntax':'passed'},indent=2),encoding='utf-8')
sys.exit(any(r['returncode'] for r in results))
