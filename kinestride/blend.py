"""Load a .blend as MJCF without a manual export step: headless Blender (iTools bl) runs the pinned blender-mjcf add-on
(Balint-H/blender-mjcf, MIT) once per file content, cached under .work/blend/<sha256>/. Other paths pass through."""
import base64, hashlib, json, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHA = '3fb51ccfe559f66714601180f8268f8587e442af'
PLUGIN = ROOT / f'.work/blender/mjcf_export_plugin-{SHA}.py'


def plugin():
    if not PLUGIN.exists() or not PLUGIN.stat().st_size:
        r = subprocess.run(['gh', 'api', f'repos/Balint-H/blender-mjcf/contents/mjcf_export_plugin.py?ref={SHA}'],
                           capture_output=True, text=True, check=True)
        PLUGIN.parent.mkdir(parents=True, exist_ok=True)
        PLUGIN.write_bytes(base64.b64decode(json.loads(r.stdout)['content']))
    return PLUGIN


def resolve(path):
    p = Path(path)
    if p.suffix != '.blend':
        return path
    xml = ROOT / '.work/blend' / hashlib.sha256(p.read_bytes()).hexdigest() / f'{p.stem}.xml'
    if not xml.exists():
        xml.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([os.environ.get('BL', str(Path.home() / 'iTools/bin/bl')), 'run', str(ROOT / 'tools/blend_export.py'), '--', str(p.resolve()), str(plugin()), str(xml)],
                           capture_output=True, text=True)
        if r.returncode or not xml.exists():
            raise SystemExit(f'FAIL blend export: {p}\n{r.stdout[-800:]}{r.stderr[-800:]}')
    return str(xml)
