import os
import re


def strip(x):
    return re.sub(
        re.compile(r'\s+'), ' ', x).replace('Inputs: ', '').rstrip('\n')


print("| Command | Description | Inputs | Outputs |")
print("| ------- | ----------- | ------ | ------- |")
for script in sorted(os.listdir(".")):
    if not script.startswith("_"):
        with open(script) as f:
            desc, io = f.read().split('"""')[1].split('\n\n')
            ii, oo = io.split('Outputs: ')
            print("| `{}` | {} | {} | {} |".format(
                script.split('.')[0], desc, strip(ii), strip(oo)))
