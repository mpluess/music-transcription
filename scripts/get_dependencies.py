from glob import glob
import re
from warnings import warn

import_pattern = re.compile(r'^\s*import (.+)')
from_pattern = re.compile(r'^\s*from ([^\s]+)')
split_pattern = re.compile(r', ?')
as_pattern = re.compile(r' as [^\s]+')
modules = set()
for file in sorted(glob('../**/*.py', recursive=True)):
    print(file)
    with open(file) as f:
        for line in f:
            if 'import ' in line and not line.startswith('#'):
                import_match = import_pattern.search(line)
                from_match = from_pattern.search(line)
                if import_match and from_match:
                    raise ValueError()
                elif import_match:
                    for module in split_pattern.split(as_pattern.sub('', import_match.group(1))):
                        modules.add(module)
                elif from_match:
                    modules.add(from_match.group(1))
                else:
                    warn('No match for line: ' + line)

for module in sorted(modules):
    if not module.startswith('music_transcription'):
        print('import ' + module)
