import random
import shutil
import os
from math import log
from tensorboard.backend.event_processing import event_accumulator
from copy import deepcopy
from time import time as curr_time
from datetime import datetime, timedelta
from Genotype import Genotype, DNA
from TrainingTask import TrainingTask


class NeuroEvolver:

    def __init__(self, reduced_population, max_population, mutation_split,
                 base_path, parameters, grammar_file):
        self.steps = 100
        self.time_cutoff = 0.1

        self.reduced_population = reduced_population
        self.max_population = max_population
        self.mutation_split = mutation_split

        self.base_path = base_path
        self.parameters = parameters
        g = Genotype(grammar_file)
        self.codon_limits = g.generate_codon_limits()

    def get_dna_from_organism(self, base_path, organism_name):
        dna_path = os.path.join(base_path, organism_name, organism_name + "_dna.txt")
        if os.path.exists(dna_path):
            dna = DNA(self.codon_limits)
            dna.load_file(dna_path)
            return dna
        else:
            return None

    def get_best_from_generation(self, generation):
        base_path = self.base_path + ("generation_%d/" % generation)
        dnas = list()
        with open(base_path+"best.txt") as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("\t"):
                    continue
                if line == "":
                    continue
                dna = self.get_dna_from_organism(base_path, line)
                dnas.append(dna)
        return dnas

    @staticmethod
    def pick_two(items):
        if len(items) == 0:
            return None
        elif len(items) == 1:
            print("only one item")
            return items[0], items[0]
        length = len(items)
        index1 = random.randint(0, length-1)
        index2 = random.randint(0, length-2)
        if index2 >= index1:
            index2 += 1
        return items[index1], items[index2]
        pass

    def preliminary_task_check(self, task):
        print("Performing preliminary check on task: %s" % task.name)
        time = task.pass_through_time(passes=16)
        print("\tPass through time: %f" % time)
        num_parameters = int(task.num_parameters())
        print("\tNumber of parameters: %f m" % (num_parameters/1000000.0))
        # flops = task.flops()
        # print("\tNumber of FLOPS: %f m" % (flops/1000000.0))
        if time < self.time_cutoff:
            print("\t\tPASS")
            return True
        else:
            print("\t\tFAIL")
            return False

    @staticmethod
    def get_losses(organism_path, event_type):
        tf_event_path = os.path.join(organism_path, "train_progress", event_type)
        if not os.path.exists(tf_event_path):
            return None
        events = os.listdir(tf_event_path)
        events = [tf_event for tf_event in events if tf_event.startswith("events")]
        if len(events) == 0:
            return None
        event = events[0]
        tf_event_path = os.path.join(tf_event_path, event)
        ea = event_accumulator.EventAccumulator(tf_event_path, size_guidance={
            event_accumulator.SCALARS: 0
        })
        ea.Reload()
        tags = ea.Tags()
        if "scalars" not in tags:
            print("Can't find scalars")
            return None
        scalars = tags["scalars"]
        if "sigmoid_loss" not in scalars:
            print("Can't find sigmoid loss")
            return None
        losses = ea.Scalars("sigmoid_loss")
        return losses

    def get_evaluations(self, generation):
        evaluations = list()
        base_path = os.path.join(self.base_path, ("generation_%d" % generation))
        organisms = [o for o in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, o))]
        for organism in organisms:
            organism_path = os.path.join(base_path, organism)
            train_losses = self.get_losses(organism_path, "train")

            train_losses = [(loss.step, loss.value) for loss in train_losses]
            train_losses.sort(key=lambda x: x[0])

            validation_losses = self.get_losses(organism_path, "validation")
            validation_losses = [(loss.step, loss.value) for loss in validation_losses]
            validation_losses.sort(key=lambda x: x[0])

            evaluations.append(((train_losses, validation_losses), organism))
        return evaluations

    def has_finished_training(self, path):
        losses = self.get_losses(path, "train")
        if losses is None:
            return False
        steps = [loss.step for loss in losses]
        if self.steps not in steps:
            return False
        return True

    def generate_new_population_tasks(self, best_dnas, generation):
        base_path = os.path.join(self.base_path, ("generation_%d" % generation))
        tasks = list()
        for i, dna in enumerate(best_dnas, 1):
            organism_name = "organism_orig_%d" % i
            training_task_path = os.path.join(base_path, organism_name)
            if os.path.isdir(training_task_path):
                if self.has_finished_training(training_task_path):
                    continue
            task = TrainingTask(dna, self.parameters, base_path, organism_name)
            tasks.append(task)
        for i, dna in enumerate(best_dnas, 1):
            mutated_originals = list()
            mutated_count = 3
            while len(mutated_originals) < mutated_count:
                organism_name = "organism_orig_%d_mutated_%d" % (i, len(mutated_originals))
                training_task_path = os.path.join(base_path, organism_name)
                if os.path.isdir(training_task_path):
                    if self.has_finished_training(training_task_path):
                        mutated_count -= 1
                        continue

                dna_copy = self.get_dna_from_organism(base_path, organism_name)
                if dna_copy is not None:
                    task = TrainingTask(dna_copy, self.parameters, base_path, organism_name)
                    mutated_originals.append(task)
                    continue
                dna_copy = deepcopy(dna)
                dna_copy.mutate(0)
                task = TrainingTask(dna_copy, self.parameters, base_path, organism_name)
                passed = self.preliminary_task_check(task)
                if passed:
                    mutated_originals.append(task)
                else:
                    task_base_path = task.file_manager.base_path
                    shutil.rmtree(task_base_path)
                    continue
            tasks.extend(mutated_originals)
        for (mutation_rate, mutation_split) in enumerate(self.mutation_split, 0):
            population_mutation_size = int(self.max_population * mutation_split)
            mutated_population_tasks = list()
            while len(mutated_population_tasks) < population_mutation_size:
                organism_name = "organism_%d_mutation_%d" % (len(mutated_population_tasks), mutation_rate)
                training_task_path = os.path.join(base_path, organism_name)
                if os.path.isdir(training_task_path):
                    if self.has_finished_training(training_task_path):
                        population_mutation_size -= 1
                        continue

                new_dna = self.get_dna_from_organism(base_path, organism_name)
                if new_dna is not None:
                    task = TrainingTask(new_dna, self.parameters, base_path, organism_name)
                    mutated_population_tasks.append(task)
                    continue
                dna_a, dna_b = self.pick_two(best_dnas)
                new_dna = dna_a.cross_over(dna_b)
                new_dna.mutate(mutation_rate)
                task = TrainingTask(new_dna, self.parameters, base_path, organism_name)
                passed = self.preliminary_task_check(task)
                if passed:
                    mutated_population_tasks.append(task)
                else:
                    task_base_path = task.file_manager.base_path
                    shutil.rmtree(task_base_path)
                    continue
            tasks.extend(mutated_population_tasks)
        tasks.reverse()
        return tasks

    def artificial_selection(self, evaluations, generation):
        norm = 0.5

        def normalize(values):
            loss_steps = values[0][0]
            losses = [loss[1] for loss in loss_steps]
            count = int(5.0*log(self.steps))
            last_few_losses = losses[-count:]
            squares = [loss**norm for loss in last_few_losses]
            return (sum(squares)/len(squares)) ** (1.0/norm)
        
        generation_path = "generation_%d/" % generation
        evaluations.sort(key=normalize)
        for evaluation in evaluations[:self.reduced_population]:
            name = evaluation[1]
            with open(self.base_path + generation_path + "best.txt", "a") as f:
                f.write(name + "\n" + "\t" + str(normalize(evaluation)) + "\n")

    def train_generation(self, generation, steps=100, time_cutoff=0.1):
        self.steps = steps
        self.time_cutoff = time_cutoff
        # get best performing organisms from previous generation
        best_dnas = self.get_best_from_generation(generation - 1)
        # get the new training tasks
        print("Creating generation %d" % generation)
        new_population_tasks = self.generate_new_population_tasks(best_dnas, generation)

        elapsed_hours_history = list()
        start_time = curr_time()

        for i, task in enumerate(new_population_tasks, 1):
            print("Training generation: %d, organism: %s" % (generation, task.name))
            pass_through_time = task.pass_through_time(passes=4)
            print("\tPass through time: %f" % pass_through_time)
            num_parameters = task.num_parameters()
            print("\t%d parameters" % num_parameters)
            # flops = task.flops()
            # print("\t%d FLOPS", flops)

            train_history = task.train(steps, verbose=False)
            train_losses = train_history[0]
            validation_losses = train_history[1]
            print("\tLoss progress:\n\t\t%s,\n\tValidation progress:\n\t\t%s" %
                  (train_losses[-5:], validation_losses[-5:]))

            elapsed_sec = curr_time() - start_time
            elapsed_hours = elapsed_sec / 60.0 / 60.0
            elapsed_hours_history.append(elapsed_hours)
            total_estimated_hours = pred_total_time(elapsed_hours_history, len(new_population_tasks))
            estimated_hours_remaining = total_estimated_hours - elapsed_hours
            now = datetime.now()
            completion_time = now + timedelta(hours=estimated_hours_remaining)
            print("Progress: %d/%d, %d hours %d minutes passed, %d hours %d minutes remaining ~ %s" %
                  (i, len(new_population_tasks),
                   int(elapsed_hours), int((elapsed_hours % 1) * 60.0),
                   int(estimated_hours_remaining), int((estimated_hours_remaining % 1) * 60.0),
                   completion_time.strftime("%I:%M %p")))
            print("-"*80)

        print("Evaluating")
        evaluations = self.get_evaluations(generation)
        self.artificial_selection(evaluations, generation)
        print("Finished evaluation")


def time_average(items):
    weights = list(range(1, len(items) + 1))
    weights = [weight ** 1.0/3.0 for weight in weights]
    weighted_sum = [x*weight for (x, weight) in zip(items, weights)]
    return sum(weighted_sum) / sum(weights)


def pred_total_time(items, pred_x):
    # https://www.varsitytutors.com/hotmath/hotmath_help/topics/line-of-best-fit
    # predict using the line of best fit
    if len(items) == 1:
        y = items[0]
        return y*pred_x
    x_bar = (len(items) + 1) / 2.0
    y_bar = sum(items) / len(items)
    m_sum_num = 0.
    m_sum_den = 0.
    for y, x in enumerate(items, 1):
        m_sum_num += (x - x_bar) * (y - y_bar)
        m_sum_den += (x - x_bar) ** 2
    m = m_sum_num / m_sum_den
    b = y_bar - m*x_bar
    return m*pred_x+b
