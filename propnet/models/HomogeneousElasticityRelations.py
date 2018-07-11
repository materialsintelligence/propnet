from propnet.core.models import AbstractModel

class HomogeneousElasticityRelations(AbstractModel):
    def plug_in(self, symbol_values):
        """
        Performs the evaluation and computational logic to accommodate the homogeneous elasticity relations.
        1) Calculates the Elastic modulus and the Shear modulus if either is un-available.
        2) Calculates the remaining unknown properties based on the Elastic and Shear moduli.
        Args:
            symbol_values (dict): dictionary containing symbols mapped to floats.
        Returns:
            (dict): mapping from string symbol to float value giving result of applying the model to the
                    given inputs.
        """
        outputs = dict()
        y = symbol_values.get("Y")
        g = symbol_values.get("G")
        if not y:
            y = HomogeneousElasticityRelations.calc_Y(symbol_values)
            outputs["Y"] = y
        if not g:
            g = HomogeneousElasticityRelations.calc_G(y, symbol_values)
            outputs["G"] = g
        remainder = HomogeneousElasticityRelations.get_constants_from_Y_G(y, g)
        for k in remainder.keys():
            if k not in symbol_values.keys():
                outputs[k] = remainder[k]
        return outputs

    @staticmethod
    def calc_Y(symbol_values):
        if "K" in symbol_values.keys():
            if "l" in symbol_values.keys():
                k = symbol_values["K"]
                l = symbol_values["l"]
                return 9*k*(k - l) / (3*k - l)
            elif "G" in symbol_values.keys():
                k = symbol_values["K"]
                g = symbol_values["G"]
                return 9*k*g / (3*k+g)
            elif "v" in symbol_values.keys():
                k = symbol_values["K"]
                v = symbol_values["v"]
                return 3 * k * (1 - 2*v)
            elif "M" in symbol_values.keys():
                k = symbol_values["K"]
                m = symbol_values["M"]
                return 9*k*(m-k)/(3*k+m)
            else:
                raise Exception("Missing required inputs to evaluate Young's Modulus")
        elif "l" in symbol_values.keys():
            if "G" in symbol_values.keys():
                l = symbol_values["l"]
                g = symbol_values["G"]
                return g*(3*l + 2*g)/(l+g)
            elif "v" in symbol_values.keys():
                l = symbol_values["l"]
                v = symbol_values["v"]
                return l*(1+v)*(1-2*v)/v
            elif "M" in symbol_values.keys():
                l = symbol_values["l"]
                m = symbol_values["M"]
                return (m-l)*(m+2*l)/(m+l)
            else:
                raise Exception("Missing required inputs to evaluate Young's Modulus")
        elif "G" in symbol_values.keys():
            if "v" in symbol_values.keys():
                g = symbol_values["G"]
                v = symbol_values["v"]
                return 2*g*(1+v)
            elif "M" in symbol_values.keys():
                g = symbol_values["G"]
                m = symbol_values["M"]
                return g*(3*m-4*g)/(m-g)
            else:
                raise Exception("Missing required inputs to evaluate Young's Modulus")
        elif "v" in symbol_values.keys():
            if "M" in symbol_values.keys():
                v = symbol_values["v"]
                m = symbol_values["M"]
                return m*(1+v)*(1-2*v)/(1-v)
            else:
                raise Exception("Missing required inputs to evaluate Young's Modulus")
        else:
            raise Exception("Missing required inputs to evaluate Young's Modulus")

    @staticmethod
    def calc_G(y, symbol_values):
        """
        Calculates the shear modulus from the provided modulus and another elastic quantity in the
        provided symbol_values dictionary.
        Returns:
            (float): value of the shear modulus or None if the calculation failed.
        """
        if "K" in symbol_values.keys():
            k = symbol_values["K"]
            return 3*k*y/(9*k - y)
        elif "l" in symbol_values.keys():
            l = symbol_values["l"]
            return (y - 3*l + (y**2 + 9*l**2 + 2*y*l)**(0.5))/4
        elif "v" in symbol_values.keys():
            v = symbol_values["v"]
            return y / (2*(1+v))
        elif "M" in symbol_values.keys():
            m = symbol_values["M"]
            return (3*m + y - (y**2 + 9*m**2 - 10*y*m)**(0.5))/8
        else:
            raise Exception("Missing required inputs to evaluate Shear Modulus")

    @staticmethod
    def get_constants_from_Y_G(y, g):
        """
        Calculates the remaining elastic quantities given the young's modulus (y),
        and the shear modulus (g)
        Returns:
             (dict): mapping from string symbol to float value
        """
        # Bulk modulus
        b = y*g / (3*(3*g-y))
        # Poisson Ratio
        v = y / (2*g) - 1
        # Lame's First Parameter
        l = g * (y - 2*g) / (3*g - y)
        # P-Wave Modulus
        p = g * (4*g - y) / (3*g - y)
        return {"K": b, "v": v, "l": l, "M": p}