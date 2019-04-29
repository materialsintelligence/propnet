import aflow.caster
import aflow.entries
import aflow.keywords
import numpy as np


# THIS MODULE IS TEMPORARY
# These are changes that should (probably) be made into a PR for the aflow library
# but for the sake of expediency, doing it this way.

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


def _ldau_TLUJ(value):
    parts = value.split(';')
    if len(parts) != 4:
        return {'ldau_params': value}
    t, l, u, j = parts
    t = aflow.caster._number(t)
    l = _numbers(l)
    u = _numbers(u)
    j = _numbers(j)

    return {
        "LDAUTYPE": t,
        "LDAUL": l,
        "LDAUU": u,
        "LDAUJ": j
    }


aflow.keywords._ldau_TLUJ.ptype = dict


class _ael_elastic_anisotropy(aflow.keywords.Keyword):
    name = "ael_elastic_anisotropy"
    ptype = float
    atype = "number"

aflow.keywords.ael_elastic_anisotropy = _ael_elastic_anisotropy()
aflow.keywords._ael_elastic_anisotropy = _ael_elastic_anisotropy
aflow.keywords._find_all()


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


def ptype(atype, keyword):
    castmap = {
        "ldau_TLUJ": "dict"
    }

    if keyword in castmap:
        return castmap[keyword]
    elif atype in castmap:
        return castmap[atype]
    else:
        return old_funcs['aflow.caster.ptype'](atype, keyword)


def cast(atype, keyword, value):
    castmap = {
        "ldau_TLUJ": _ldau_TLUJ
    }

    if value is None:
        return None

    if keyword in castmap:
        return castmap[keyword](value)
    elif atype in castmap:
        return castmap[atype](value)
    else:
        return old_funcs['aflow.caster.cast'](atype, keyword, value)


old_funcs = {
    'aflow.caster._numbers': aflow.caster._numbers,
    'aflow.caster._kpoints': aflow.caster._kpoints,
    'aflow.caster.ptype': aflow.caster.ptype,
    'aflow.caster.cast': aflow.caster.cast,
    'aflow.entries._val_from_str': aflow.entries._val_from_str
}

aflow.caster._numbers = _numbers
aflow.caster._kpoints = _kpoints
aflow.caster.ptype = ptype
aflow.caster.cast = cast
aflow.entries._val_from_str = _val_from_str
aflow.caster.exceptions.append('ldau_TLUJ')