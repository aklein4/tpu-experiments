
import importlib


def import_class(path, *args):
    dot = path.rfind('.')
    module_name = path[:dot]
    class_name = path[dot + 1:]
    
    for a in args[::-1]:
        module_name = f"{a}.{module_name}"

    # import the module
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(f'Could not import module {module_name}')

    # import the class
    if hasattr(module, class_name):
        class_ = getattr(module, class_name)
    else:
        raise ImportError(f'Could not find class {class_name} in module {module_name}.')
    
    print(f"Imported class {class_name} from module {module_name}", flush=True)
    return class_