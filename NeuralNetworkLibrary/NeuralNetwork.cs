using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class NeuralNetwork
    {
        public Layer[] Layers;
        private ErrorFunction errorFunc;

        public NeuralNetwork(ActivationFunction activation, ErrorFunction errorFunc, params int[] neuronsPerLayer)
        {
            Layers = new Layer[neuronsPerLayer.Length];
            for(int i = 0; i < neuronsPerLayer.Length; i++)
            {
                Layer prev = new Layer(activation, 0, null);
                if (i - 1 >= 0)
                {
                    prev = new Layer(activation, neuronsPerLayer[i - 1], Layers[i]);
                    prev = Layers[i - 1];
                }

                Layers[i] = new Layer(activation, neuronsPerLayer[i], prev);
            }
            this.errorFunc = errorFunc;
        }

        public void Randomize(Random random, int min, int max) 
        {
            for(int i = 0; i < Layers.Length; i++)
            {
                Layers[i].Randomize(random, min, max);
            }
        }

        public double[] Compute(double[] inputs) 
        {
            Layers[0].SetOutputs(inputs);

            for(int i = 1; i < Layers.Length; i++)
            {
                Layers[i].Compute();
            }

            return Layers[Layers.Length - 1].Outputs;
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
