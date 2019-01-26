"""
Module containing metaclasses for the registry pattern in propnet
This also enforces a single instance per name
"""

import abc


class RegistryMeta(abc.ABCMeta):
    def __init__(cls, name, bases, nmspc):
        super(RegistryMeta, cls).__init__(name, bases, nmspc)

        # This ensures we have a master registry at the root class so anything
        # inheriting from Model will register to Model
        cls.all_instances = None
        for b in bases:
            if isinstance(b, RegistryMeta):
                cls.all_instances = b.all_instances
                break

        # If we didn't find master registry then make a new one
        if cls.all_instances == None:
            cls.all_instances = dict()

    def __call__(cls, *args, **kwargs):

        # First get the name
        if "name" in kwargs:
            name = kwargs["name"]
        elif len(args) < 1:
            raise ValueError("Can't get a propnet object without a name")
        else:
            name = args[0]

        # If there isn't already an instance with this name
        # Let's make a new object
        if name in cls.all_instances:
            if getattr(cls, "fail_on_duplicate", False):
                raise ValueError(f"Can't instantiate more than one of {cls.__name__} named {name}")
            else:
                return cls.all_instances[name]
        else:
            new_object = super(RegistryMeta, cls).__call__(*args, **kwargs)
            cls.all_instances[name] = new_object

        return cls.all_instances[name]

    def __getitem__(cls, name):
        if name in cls.all_instances:
            return cls.all_instances[name]
        else:
            raise ValueError("No instances of {cls.__name__} named {name}")
