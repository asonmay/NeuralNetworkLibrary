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
        public double[] Outputs { get; private set; }

        public Layer(ActivationFunction activation, int neuronCount, Layer previousLayer) 
        {
            Neurons = new Neuron[neuronCount];
            for(int i = 0; i < Neurons.Length; i++)
            {
                if(previousLayer == null)
                {
                    Neurons[i] = new Neuron(activation, []);
                }
                else
                {
                    Neurons[i] = new Neuron(activation, previousLayer.Neurons);
                }
            }
            Outputs = new double[neuronCount];
            Outputs = Compute();
        }

        public void ApplyChanges(double momentum)
        {
            for(int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].ApplyChanges(momentum);
            }
        }

        public void SetOutputs(double[] values)
        {
            Outputs = values;
            for(int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].Output = Outputs[i];
            }
        }

        public void Randomize(Random random, double min, double max) 
        {
            for(int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].Randomize(random, min, max);
            }
        }

        public void BackProp(double learningRate)
        {
            for(int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].BackProp(learningRate);
            }
        }

        public double[] Compute() 
        {
            for(int i = 0; i < Outputs.Length; i++)
            {
                Outputs[i] = Neurons[i].Compute();
                Neurons[i].Output = Outputs[i];
            }
            return Outputs;
        }
    }
}
