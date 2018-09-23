from Genotype import Genotype, DNA
from TrainingTask import TrainingTask

base_path = "/Users/Yuhan/Documents/DocumentsMacBook/Programming/_Projects/Experimental/AI/EvolutionaryNN/networks/"
base_path += "deep_cnn/"
training_data_path = "/Users/Yuhan/Documents/DocumentsMacBook/Programming/_Projects/Experimental/HandTracking/"
training_data_path += "_DataSets/RHD_published_v2/"
parameters = {
    "color_files": training_data_path + "training/color/",
    "mask_files": training_data_path + "training/mask/",
    "eval_color_files": training_data_path + "evaluation/color/",
    "eval_mask_files": training_data_path + "evaluation/mask/",
    "cache_location_base": training_data_path + "cached/"
}
grammar_file = base_path + "network_grammar.txt"
g = Genotype(grammar_file)
codon_limits = g.generate_codon_limits()

training_path = "./networks/deep_cnn/"
dna = DNA(codon_limits)
dna.load_file(training_path + "dna.txt")

task = TrainingTask(dna, parameters, base_path, "v2")
print(task.pass_through_time(passes=4))
task.train(1000, step=0, verbose=True)
