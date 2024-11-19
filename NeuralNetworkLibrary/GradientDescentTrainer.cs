using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class GradientDescentTrainer
    {
        public NeuralNetwork Network;
        private double[][] inputs;
        private double[][] desiredOutputs;
        private ActivationFunction activationFunction;

        public GradientDescentTrainer(double[][] inputs, double[][]desiredOutputs, ActivationFunction activation, ErrorFunction error, double min, double max, params int[] neuronsPerLayer)
        {
            this.inputs = inputs;
            this.desiredOutputs = desiredOutputs;
            Network = new NeuralNetwork(activation, error, neuronsPerLayer);
            Network.Randomize(new Random(), min, max);
            activationFunction = activation;
        }

        public double Train(double learningRate, double currentError, double momentum, int batchSize)
        {
            for(int i = 0; i < batchSize; i++)
            {
                for (int x = i * batchSize; x < (i + 1) * batchSize; x++)
                {
                    Network.Compute(inputs[x]);
                    Network.BackProp(GetLearningRate(currentError, learningRate), desiredOutputs[x], inputs[x]);
                }
                Network.ApplyChanges(momentum);
            } 

            double error = Network.GetError(inputs,desiredOutputs);
            Console.SetCursorPosition(0, 0);
            Console.Write($"Error: {error}");

            return error;
        }

        private double GetLearningRate(double error, double original)
        {
            return original * error;
        }

        public void Train(double learningRate, float error, double momentum, int batchSize)
        {
            double currentError = Network.GetError(inputs, desiredOutputs);
            while (currentError > error)
            { 
                currentError = Train(learningRate, currentError, momentum, batchSize);
                for(int i = 0; i < inputs.Length; i++)
                {
                    Console.SetCursorPosition(0, i + 1);
                    Console.Write("the sine of ");
                    Console.Write(inputs[i][0]);
                    Console.Write(" is ");
                    Console.Write(Network.Compute(inputs[i])[0]);
                }
            }
        }
    }
}
