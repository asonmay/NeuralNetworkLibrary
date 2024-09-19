using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class NeuralNetwork
    {
        private Layer[] layers;
        private ErrorFunction errorFunc;

        public NeuralNetwork(ActivationFunction activation, ErrorFunction errorFunc, params int[] neuronsPerLayer)
        {
            layers = new Layer[neuronsPerLayer.Length];
            for(int i = 0; i < neuronsPerLayer.Length; i++)
            {
                Layer prev = new Layer(activation, layers[i].Neurons.Length, layers[i]);
                if (i - 1 >= 0)
                {
                    prev = layers[i - 1];
                }
                layers[i] = new Layer(activation, neuronsPerLayer[i], prev);
            }
            this.errorFunc = errorFunc;
        }

        public void Randomize(Random random, int min, int max) 
        {
            for(int i = 0; i <= layers.Length; i++)
            {
                layers[i].Randomize(random, min, max);
            }
        }

        public double[] Compute(double[] inputs) 
        {
            layers[0].SetOutputs(inputs);

            for(int i = 1; i <  layers.Length; i++)
            {
                layers[i].Compute();
            }

            return layers[layers.Length].Outputs;
        }

        public double GetError(double[] inputs, double[] desiredOutputs) 
        {
            double[] output = Compute(inputs);
            double[] error = new double[output.Length];
            for(int i = 0; i < output.Length; i++)
            {
                error[i] = errorFunc.Function(output[i], desiredOutputs[i]);
            }

            return error.Sum();
        }
    }
}
