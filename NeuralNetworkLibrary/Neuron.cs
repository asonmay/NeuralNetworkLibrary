using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics.Arm;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class Neuron
    {
        private double bias;
        private Dendrite[] dendrites;
        public double Output { get; set; }
        public double Input { get; private set; }
        public ActivationFunction Activation { get; set; }

        public Neuron(ActivationFunction activation, Neuron[] previousNerons) 
        {
            Activation = activation;
            dendrites = new Dendrite[previousNerons.Length];
            for(int i = 0; i < previousNerons.Length; i++)
            {
                dendrites[i] = new Dendrite(previousNerons[i], this, 1);
            }
        }

        public void Randomize(Random random, int min, int max) 
        {
            bias = random.Next(min, max);
            for(int i = 0; i < dendrites.Length; i++)
            {
                dendrites[i].Weight = random.Next(min, max);
            }
        }

        public double Compute() 
        {
            double sum = 0;
            for(int i = 0; i <= dendrites.Length; i++)
            {
                sum += dendrites[i].Compute();
            }
            return sum + bias;
        }
    }
}
