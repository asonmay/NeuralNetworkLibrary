using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class GeneticTrainer
    {
        private NeuralNetwork[] networks;

        private double mutationRate;
        private int min;
        private int max;

        public GeneticTrainer(int min, int max)
        {
            this.min = min;
            this.max = max;
        }

        private void MutateWeights(Neuron neuron, Random random)
        {
            for (int i = 0; i < neuron.Dendrites.Length; i++)
            {
                if (random.NextDouble() < mutationRate)
                {
                    int randomNum = random.Next(3);
                    if (randomNum == 0)
                    {
                        neuron.Bias += 1;
                    }
                    else if (randomNum == 1)
                    {
                        neuron.Bias -= 1;
                    }
                    else
                    {
                        neuron.Dendrites[i].Weight *= -1;
                    }
                }
            }
        }

        private void MutateBias(Neuron neuron, Random random)
        {
            if (random.NextDouble() < mutationRate)
            {
                int randomNum = random.Next(3);
                if (randomNum == 0)
                {
                    neuron.Bias += 1;
                }
                else if (randomNum == 1)
                {
                    neuron.Bias -= 1;
                }
                else
                {
                    neuron.Bias *= -1;
                }
            }
        }

        public void Mutate(NeuralNetwork net, Random random)
        {
            foreach (Layer layer in net.Layers)
            {
                foreach (Neuron neuron in layer.Neurons)
                {
                    MutateWeights(neuron, random);
                    MutateBias(neuron, random);
                }
            }
        }

        public void Crossover(NeuralNetwork winner, NeuralNetwork loser, Random random)
        {
            for (int i = 0; i < winner.Layers.Length; i++)
            {
                Layer winLayer = winner.Layers[i];
                Layer childLayer = loser.Layers[i];

                int cutPoint = random.Next(winLayer.Neurons.Length);
                bool flip = random.Next(2) == 0;

                for (int j = (flip ? 0 : cutPoint); j < (flip ? cutPoint : winLayer.Neurons.Length); j++)
                {
                    Neuron winNeuron = winLayer.Neurons[j];
                    Neuron childNeuron = childLayer.Neurons[j];

                    CopyWeights(childNeuron, winNeuron);
                    childNeuron.Bias = winNeuron.Bias;
                }
            }
        }

        private void CopyWeights(Neuron childNeuron, Neuron winNeuron)
        {
            for(int i = 0; i < childNeuron.Dendrites.Length; i++)
            {
                childNeuron.Dendrites[i].Weight = winNeuron.Dendrites[i].Weight;
            }
        }

        public void Train((NeuralNetwork net, double fitness)[] population, Random random, double mutationRate)
        {
            Array.Sort(population, (a, b) => b.fitness.CompareTo(a.fitness));

            int start = (int)(population.Length * 0.1);
            int end = (int)(population.Length * 0.9);

            for (int i = start; i < end; i++)
            {
                Crossover(population[random.Next(start)].net, population[i].net, random);
                Mutate(population[i].net, random);
            }

            for (int i = end; i < population.Length; i++)
            {
                population[i].net.Randomize(random, min, max);
            }
        }
    }
}
