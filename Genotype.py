from collections import OrderedDict
import random
from NetworkLayers import NetworkLayers as Net
from copy import deepcopy


def bool_cvt(string):
    string = string.lower()
    if string == "false":
        return False
    elif string == "true":
        return True
    else:
        raise ValueError("invalid literal for bool: %s", string)


class NucleotideLimits:

    def __init__(self, layer_type, parameters):
        self.layer_type = layer_type
        self.parameters = parameters

    def __str__(self):
        return "Nucleotide type: '" + self.layer_type + "' " +\
               "Parameters: " + self.parameters.__str__()

    def __repr__(self):
        return self.__str__()


class Genotype:

    def __init__(self, grammar_file):
        self.grammar_file = grammar_file

    def generate_codon_limits(self):
        """Generates the nucleotide limits within each codon"""
        with open(self.grammar_file) as f:
            codon_limits = OrderedDict()
            codon_name = None
            layer_name = None
            settings = dict()
            for line in f:
                line = line.rstrip()
                if line == "":
                    continue
                if line.startswith("--"):
                    codon_name = line[2:-2]
                elif line.startswith("\t"):
                    setting = line.split(":")
                    setting_name = setting[0]
                    setting_config = setting[1]
                    # remove all whitespace
                    setting_name = "".join(setting_name.split())
                    setting_config = "".join(setting_config.split())
                    settings[setting_name] = setting_config
                elif line.endswith("{"):
                    layer_name = line[:-2]
                    settings = dict()
                elif line == "}":
                    nucleotide_limit = NucleotideLimits(layer_name, settings)
                    if codon_name not in codon_limits:
                        codon_limits[codon_name] = list()
                    codon_limits[codon_name].append(nucleotide_limit)
        return codon_limits


class Nucleotide:

    def __init__(self):
        self.type = "Unintialized"
        self.parameters = dict()

    def perturb(self, mutation_rate, parameters):
        sigma_width = 9.0
        change_rate = 1.0
        if mutation_rate == 1:
            sigma_width = 6.0
            change_rate = 0.8
        if mutation_rate == 2:
            sigma_width = 3.0
            change_rate = 0.9
        if mutation_rate == 3:
            sigma_width = 2.0
            change_rate = 1.0
        if random.uniform(0, 1) > change_rate:
            # don't perturb anything
            # return
            pass
        for k, v in parameters.items():
            curr_val = self.parameters[k]
            opts = v.split(",")
            val_type = opts[0]
            func = look_up[val_type]
            new_val = func(*(opts[1:]+[curr_val, sigma_width]))
            self.parameters[k] = new_val

    def load_from_parameters(self, layer_type, parameters):
        self.type = layer_type
        self.parameters = parameters

    def generate_from_limits(self, limits):
        self.type = limits.layer_type
        self.parameters.clear()
        parameters = limits.parameters
        for setting, limit in parameters.items():
            self.parameters[setting] = get_val_from_limit(limit)

    def referenced_copy(self):
        copy = Nucleotide()
        copy.type = self.type
        copy.parameters = self.parameters
        return copy

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Type: '" + self.type + "' " +\
                self.parameters.__str__()


class Codon:

    def __init__(self, name, nucleotide_limits):
        self.name = name
        self.nucleotide_limits = nucleotide_limits
        self.nucleotides = list()

    def mutate(self, mutation_rate):
        rate = 0.0
        if mutation_rate == 1:
            rate = 0.1
        elif mutation_rate == 2:
            rate = 0.3
        elif mutation_rate == 3:
            rate = 0.5
        # random removals
        for index in range(len(self.nucleotides)-1, -1, -1):
            if random.uniform(0, 1) < rate:
                self.nucleotides.pop(index)
        # random perturbations
        for nucleotide in self.nucleotides:
            for limit in self.nucleotide_limits:
                if limit.layer_type == nucleotide.type:
                    nucleotide.perturb(mutation_rate, limit.parameters)
                    break
        # random additions
        for index in range(len(self.nucleotides), -1, -1):
            # we want the number of additions to be on average the same
            # as the number of removals to maintain the size of the network
            new_rate = 1 / (1 - rate) - 1
            if len(self.nucleotides) == 0:
                new_rate = 0.8
            if random.uniform(0, 1) < new_rate:
                new_nucleotide = self.create_random_nucleotide()
                self.nucleotides.insert(index, new_nucleotide)
        # random reference copies
        copies = list()
        for nucleotide in self.nucleotides:
            new_rate = 1/(1-rate) - 1
            new_rate /= 8
            if random.uniform(0, 1) < new_rate:
                copies.append(nucleotide.referenced_copy())
        for copy in copies:
            index = random.randint(0, len(self.nucleotides))
            self.nucleotides.insert(index, copy)

    def pointwise_crossover(self, other):
        cutoff_point = random.uniform(-0.1, 1.1)
        cutoff_point = min(cutoff_point, 1)
        cutoff_point = max(cutoff_point, 0)
        cutoff_a = int(round(cutoff_point * len(self.nucleotides)))
        cutoff_b = int(round(cutoff_point * len(other.nucleotides)))
        new_nucleotides = self.nucleotides[:cutoff_a] + other.nucleotides[cutoff_b:]
        new_codon = Codon(deepcopy(self.name), deepcopy(self.nucleotide_limits))
        new_codon.nucleotides = deepcopy(new_nucleotides)
        return new_codon

    def add_nucleotide(self, nucleotide):
        self.nucleotides.append(nucleotide)

    def init_rand_nucleotides(self, amt):
        for i in range(amt):
            nucleotide = self.create_random_nucleotide()
            self.nucleotides.append(nucleotide)

    def create_random_nucleotide(self):
        limit = random.choice(self.nucleotide_limits)
        nucleotide = Nucleotide()
        nucleotide.generate_from_limits(limit)
        return nucleotide

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Codon: '" + self.name + "' " +\
                self.nucleotides.__str__()


class DNA:

    def __init__(self, codon_limits, codons=None):
        if type(codons) is not list:
            self.codons = list()
            for codon_name, nucleotide_limits in codon_limits.items():
                codon = Codon(codon_name, nucleotide_limits)
                self.add_codon(codon)
        else:
            self.codons = codons

    def mutate(self, mutation_rate):
        # mutation_rate:
        #   0 - basic mutation, no removal, additions, copies,
        #           just perturbations
        #   1 - simple mutations, minimal removal, additions, copies,
        #           perturbations
        #   2 - moderate mutations, minimal removal, additions, copies,
        #           moderate perturbations
        #   3 - extreme mutations, high removals, additions, copies
        #           perturbations
        for codon in self.codons:
            codon.mutate(mutation_rate)

    def cross_over(self, other):
        new_codons = list()
        cross_over = random.getrandbits(len(self.codons))
        for (codon_a, codon_b) in zip(self.codons, other.codons):
            if cross_over % 2 == 1:
                codon_a, codon_b = codon_b, codon_a
            cross_over >>= 1
            new_codon = codon_a.pointwise_crossover(codon_b)
            new_codons.append(new_codon)
        return DNA(None, codons=new_codons)
        pass

    def add_codon(self, codon):
        self.codons.append(codon)

    type_lookup = {
        "int": int,
        "float": float,
        "bool": bool_cvt
    }

    def load_file(self, file_path):
        with open(file_path) as f:
            curr_codon = None
            curr_type = None
            parameters = dict()
            for line in f:
                line = line.rstrip()
                if line == "":
                    continue
                if line.startswith("--"):
                    name = line[2:-2]
                    for codon in self.codons:
                        if codon.name == name:
                            curr_codon = codon
                            break
                elif line.startswith("\t"):
                    setting = line.split(":")
                    setting_name = setting[0]
                    setting_config = setting[1]
                    # remove all whitespace
                    setting_name = "".join(setting_name.split())
                    setting_config = "".join(setting_config.split())
                    # split type and value
                    config = setting_config.split(",")
                    config_type = DNA.type_lookup[config[0]]
                    config_value = config_type(config[1])
                    parameters[setting_name] = config_value
                elif line.endswith("{"):
                    curr_type = line[:-2]
                    parameters = dict()
                elif line == "}":
                    nucleotide = Nucleotide()
                    nucleotide.load_from_parameters(curr_type, parameters)
                    curr_codon.add_nucleotide(nucleotide)

    def save_file(self, file_path):
        with open(file_path, "w") as f:
            for codon in self.codons:
                f.write("--" + codon.name + "--\n")
                for nucleotide in codon.nucleotides:
                    f.write(nucleotide.type + " {\n")
                    for setting_name, setting_val in nucleotide.parameters.items():
                        f.write("\t")
                        f.write(setting_name + " : ")
                        setting_type = type(setting_val)
                        if setting_type is int:
                            setting_type = "int"
                        elif setting_type is float:
                            setting_type = "float"
                        elif setting_type is bool:
                            setting_type = "bool"
                        else:
                            setting_type = "unspecified"
                        f.write(setting_type + ", ")
                        f.write(str(setting_val).lower() + "\n")
                    f.write("}\n")

    def generate_from_limits(self):
        for codon in self.codons:
            codon.init_rand_nucleotides(2)


class Ribosome:

    def __init__(self):
        pass

    @classmethod
    def get_layers(cls, codons):
        layers = list()
        for codon in codons:
            layers.extend(cls._cvt_codon_to_layers(codon))
        return layers

    @classmethod
    def _cvt_codon_to_layers(cls, codon):
        layers = list()
        nucleotides = codon.nucleotides
        for nucleotide in nucleotides:
            layers.append(cls._cvt_nucleotide_to_layer(nucleotide))
        return layers

    @classmethod
    def _cvt_nucleotide_to_layer(cls, nucleotide):
        nucleotide_type = nucleotide.type
        parameters = nucleotide.parameters
        if nucleotide_type == "ConvolutionSquare":
            filter_size = parameters["filter_size"]
            out_chan = parameters["out_chan"]
            relu = parameters["relu"]

            def curried(in_tensor, layer_name, trainable):
                x = Net.conv(in_tensor, layer_name, filter_size, stride=1,
                             out_chan=out_chan, trainable=trainable)
                if relu:
                    x = Net.relu(x, name=layer_name + "_relu")
                return x

            return curried
        elif nucleotide_type == "ConvolutionRect":
            filter_size_small = parameters["filter_size_small"]
            filter_size_large = parameters["filter_size_large"]
            out_chan = parameters["out_chan"]
            relu = parameters["relu"]

            def curried(in_tensor, layer_name, trainable):
                x = Net.conv(in_tensor, layer_name+"_horizontal", [filter_size_small, filter_size_large], stride=1,
                             out_chan=out_chan, trainable=trainable)
                x = Net.conv(x, layer_name+"_vertical", [filter_size_large, filter_size_small], stride=1,
                             out_chan=out_chan, trainable=trainable)
                if relu:
                    x = Net.relu(x, name=layer_name + "_relu")
                return x

            return curried
        elif nucleotide_type == "InceptionBase":
            intermediate_chan = parameters["intermediate_chan"]

            def curried(in_tensor, layer_name, trainable):
                x = Net.inception_module_base(in_tensor, intermediate_chan,
                                              layer_name=layer_name, trainable=trainable)
                return x

            return curried
        elif nucleotide_type == "InceptionSimple":
            intermediate_chan = parameters["intermediate_chan"]

            def curried(in_tensor, layer_name, trainable):
                x = Net.inception_module_simple(in_tensor, intermediate_chan,
                                                layer_name=layer_name, trainable=trainable)
                return x

            return curried
        elif nucleotide_type == "MobileNetResidual":
            expansion_factor = parameters["expansion_factor"]
            out_chan = parameters["out_chan"]

            def curried(in_tensor, layer_name, trainable):
                x = Net.mobilenetv1_module_residual(in_tensor, layer_name, expansion_factor,
                                                    out_chan, trainable=trainable)
                return x

            return curried
        elif nucleotide_type == "MobileNetBase":
            expansion_factor = parameters["expansion_factor"]
            out_chan = parameters["out_chan"]

            def curried(in_tensor, layer_name, trainable):
                x = Net.mobilenetv1_module_base(in_tensor, layer_name, expansion_factor,
                                                out_chan, trainable=trainable)
                return x

            return curried
        elif nucleotide_type == "AveragePool":
            def curried(in_tensor, layer_name, _):
                x = Net.avg_pool(in_tensor, layer_name)
                return x

            return curried
        elif nucleotide_type == "MaxPool":
            def curried(in_tensor, layer_name, _):
                x = Net.avg_pool(in_tensor, layer_name)
                return x

            return curried
        elif nucleotide_type == "Pool":
            max_pool = parameters["max_pool"]
            if max_pool:
                def curried(in_tensor, layer_name, _):
                    x = Net.avg_pool(in_tensor, layer_name)
                    return x

                return curried
            else:
                def curried(in_tensor, layer_name, _):
                    x = Net.avg_pool(in_tensor, layer_name)
                    return x

                return curried
        elif nucleotide_type == "DeconvolutionSquare":
            filter_size = parameters["filter_size"]
            out_chan = parameters["out_chan"]
            strides = parameters["strides"]
            relu = parameters["relu"]

            def curried(in_tensor, layer_name, trainable):
                x = Net.upconv(in_tensor, layer_name, out_chan, filter_size,
                               strides, trainable=trainable)
                if relu:
                    x = Net.relu(x, name=layer_name + "_relu")
                return x

            return curried


def random_bool(mu=None, sigma_width=None):
    if mu is None:
        mu = 0.5
    if sigma_width is None:
        sigma_width = 2.0
    sigma = 4.0/sigma_width
    r = random.gauss(mu, sigma)
    r = max(r, 0)
    r = min(r, 1)
    return bool(int(round(r)))


def random_int(a, b, mu=None, sigma_width=None):
    a = float(a)
    b = float(b)
    if mu is None:
        mu = (a + b) / 2.0
    if sigma_width is None:
        sigma_width = 2.0
    sigma = (a-b) / sigma_width
    r = random.gauss(mu, sigma)
    r = max(r, a)
    r = min(r, b)
    return int(round(r))


def random_float(a, b, mu=None, sigma_width=None):
    a = float(a)
    b = float(b)
    if mu is None:
        mu = (a + b) / 2.0
    if sigma_width is None:
        sigma_width = 2.0
    sigma = (a-b) / sigma_width
    r = random.gauss(mu, sigma)
    r = max(r, a)
    r = min(r, b)
    return r


look_up = {
    "bool": random_bool,
    "int": random_int,
    "float": random_float
}


def get_val_from_limit(limit):
    opts = limit.split(",")
    val_type = opts[0]
    func = look_up[val_type]
    return func(*opts[1:])
