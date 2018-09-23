from NeuroEvolver import NeuroEvolver


base_path = "/Users/Yuhan/Documents/DocumentsMacBook/Programming/_Projects/Experimental/AI/EvolutionaryNN/networks/"
base_path += "mobile_net_segmentation/"
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

evolver = NeuroEvolver(reduced_population=15, max_population=10,
                       mutation_split=[0.4, 0.2, 0.2, 0.2],
                       base_path=base_path, parameters=parameters,
                       grammar_file=grammar_file)
evolver.train_generation(1, steps=400, time_cutoff=0.075)

