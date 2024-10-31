using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class GeneticTrainer<T> where T : INeuralNetwork
    {
        public T[] Networks;
        public int Generation;
        private double mutationRate;
        private int min;
        private int max;
        private double copyPercent;
        private double randomizePercent;

        public GeneticTrainer(int min, int max, double mutationRate, T[] networks, double copyPercent, double randomizePercent)
        {
            this.min = min;
            this.max = max;
            this.mutationRate = mutationRate;
            Networks = networks;
            this.copyPercent = copyPercent;
            this.randomizePercent = randomizePercent;
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
                        neuron.Dendrites[i].Weight += mutationRate;
                    }
                    else if (randomNum == 1)
                    {
                        neuron.Dendrites[i].Weight -= mutationRate;
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
                    neuron.Bias += mutationRate;
                }
                else if (randomNum == 1)
                {
                    neuron.Bias -= mutationRate;
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
                int count = flip ? cutPoint : winLayer.Neurons.Length;
                for (int j = (flip ? 0 : cutPoint); j < count; j++)
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

        public void SortByFitness()
        {
            Array.Sort(Networks, (a, b) => b.Fitness.CompareTo(a.Fitness));
        }

        public void Train(Random random)
        {
            Generation++;
            Array.Sort(Networks, (a, b) => b.Fitness.CompareTo(a.Fitness));

            int start = (int)(Networks.Length * copyPercent);
            int end = (int)(Networks.Length * (1 - randomizePercent));

            for (int i = start; i < end; i++)
            {
                Crossover(Networks[random.Next(start)].Network, Networks[i].Network, random);
                Mutate(Networks[i].Network, random);
            }

            for (int i = end; i < Networks.Length; i++)
            {
                Networks[i].Network.Randomize(random, min, max);
            }
        }
    }
}
