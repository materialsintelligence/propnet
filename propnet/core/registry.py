"""
Module containing metaclasses for the registry pattern in propnet
This also enforces a single instance per name
"""

import abc


class RegistryMeta(abc.ABCMeta):
    def __init__(cls, name, bases, nmspc):
        super(RegistryMeta, cls).__init__(name, bases, nmspc)
        cls.all_instances = dict()

    def __call__(cls, *args, **kwargs):

        # First get the name
        if "name" in kwargs:
            name = kwargs["name"]
        elif len(args) < 1:
            raise TypeError("Can't get a propnet object without a name")
        else:
            name = args[0]

        # If there isn't already an instance with this name
        # Let's make a new object
        if name not in cls.all_instances:
            new_object = super(RegistryMeta, cls).__call__(*args, **kwargs)
            cls.all_instances[id] = new_object

        return cls.all_instances[id]
