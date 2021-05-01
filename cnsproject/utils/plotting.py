from typing import List, Union, Dict


def param2title(parameter_set, selected: Union[List[str], Dict[str, str]]) -> str:
    result = ""
    first = True
    for s in selected:
        if not first:
            result += ", "
        else:
            first = False

        if isinstance(selected, dict):
            value = parameter_set.get(s, "")
            result += f"{selected[s]}: {value}"
        else:
            value = parameter_set.get(s, "")
            result += f"{s}: {value}"
    return result

