import aflow.caster
import aflow.entries
import numpy as np


# Redefining aflow library functions because they're not general enough
def _numbers(value):
    svals = list(value.split(','))
    vals = list(map(aflow.caster._number, svals))
    return np.array(vals)


def _kpoints(value):
    parts = value.split(';')
    relaxation = np.array(list(map(aflow.caster._number, parts[0].split(','))))
    if len(parts) == 1:
        return {"relaxation": relaxation}
    static = np.array(list(map(aflow.caster._number, parts[1].split(','))))
    if len(parts) == 3:  # pragma: no cover
        # The web page (possibly outdated) includes an example where
        # this would be the case. We include it here for
        # completeness. I haven't found a case yet that we could use in
        # the unit tests to trigger this.
        points = parts[-1].split('-')
        nsamples = None
    else:
        points = parts[-2].split('-')
        nsamples = int(parts[-1])

    return {
        "relaxation": relaxation,
        "static": static,
        "points": points,
        "nsamples": nsamples
    }


def _val_from_str(attr, value):
    """Retrieves the specified attribute's value, cast to an
    appropriate python type where possible.
    """
    clsname = "_{}".format(attr)
    if hasattr(aflow.entries.kw, clsname):
        cls = getattr(aflow.entries.kw, clsname)
        atype = getattr(cls, "atype")
        return aflow.caster.cast(atype, attr, value)
    else:
        return value


aflow.caster._numbers = _numbers
aflow.caster._kpoints = _kpoints
aflow.entries._val_from_str = _val_from_str