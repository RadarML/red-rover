import os, re


def _parse_bullets(s):
    return [
        line.strip().replace('\n      ', ' ').lstrip('- ').strip()
        for line in s.split('\n    -') if line.strip()]


def _parse_script(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    pattern = re.compile(
        r'"""(.*?)\n\n(.*?)Inputs:\n(.*?)\n\nOutputs:\n(.*?)(?:\n\n|""")',
        re.DOTALL)
    match = pattern.search(content)

    if not match:
        raise ValueError(f"Docstring does not match format: {file_path}")

    desc = match.group(1).strip()
    desc2 = re.sub(r'(?<!\n)\n(?!\n)', ' ', match.group(2).strip())
    inputs = match.group(3).strip()
    outputs = match.group(4).strip()

    return desc, desc2, _parse_bullets(inputs), _parse_bullets(outputs)


# print("| Command | Description | Inputs | Outputs |")
# print("| ------- | ----------- | ------ | ------- |")
for script in sorted(os.listdir(".")):
    if not script.startswith("_") and script.endswith('.py'):
        desc, _, inputs, outputs = _parse_script(script)
        print(f"- `{script.split('.')[0]}`: {desc}")
        print(f"   - {', '.join(inputs)} &rarr; {', '.join(outputs)}")
        # print("| `{}` | {} | {} | {} |".format(
        #     script.split('.')[0], desc, ', '.join(inputs), ', '.join(outputs)))
