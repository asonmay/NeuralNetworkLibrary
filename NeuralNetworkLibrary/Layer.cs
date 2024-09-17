using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class Layer
    {
        public Neuron[] Neurons { get; }
        public double[] Outputs { get; }

        public Layer(ActivationFunction activation, int neuronCount, Layer previousLayer) 
        {
            Neurons = new Neuron[neuronCount];
            for(int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(activation, previousLayer.Neurons);
            }
            Outputs = Compute();
        }

        public void Randomize(Random random, int min, int max) 
        {
            for(int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].Randomize(random, min, max);
            }
        }

        public double[] Compute() 
        {
            double[] results = new double[Neurons.Length];
            for(int i = 0; i < results.Length; i++)
            {
                results[i] = Neurons[i].Compute();
            }
            return results;
        }
    }
}
