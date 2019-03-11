"""
Module containing metaclasses for the registry pattern in propnet
This also enforces a single instance per name
"""

import abc


class RegistryMeta(abc.ABCMeta):

    all_instances = None

    def __init__(cls, name, bases, nmspc):
        super(RegistryMeta, cls).__init__(name, bases, nmspc)

        # If we didn't find master registry then make a new one
        if cls.all_instances is None:
            cls.all_instances = dict()

    def __call__(cls, *args, **kwargs):
        if "name" in kwargs:
            # First get the name
            name = kwargs["name"]
            kwargs.pop("name",None)
        elif len(args) > 0:
            # assume first arg is name
            name = args[0]
            args = args[1:]
        else:
            raise ValueError("Must specify a name to get a Registry")

        if name not in cls.all_instances:
            new_object = super(RegistryMeta, cls).__call__(*args, **kwargs)
            cls.all_instances[name] = new_object

        return cls.all_instances[name]

    def clear_all_registries(cls):
        cls.all_instances.clear()


class Registry(dict, metaclass=RegistryMeta):
    pass
