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

        public double Train(double learningRate, double currentError)
        {
            for(int i = 0; i < desiredOutputs.Length; i++)
            {
                Network.Compute(inputs[i]);
                Network.BackProp(GetLearningRate(currentError, learningRate), desiredOutputs[i], inputs[i]);
            }
            Network.ApplyChanges();

            double error = Network.GetError(inputs,desiredOutputs);
            Console.SetCursorPosition(0, 0);
            Console.Write($"Error: {error}");

            return error;
        }

        public void Train(double learningRate, int iterations, double currentError)
        {
            double current = 0;
            while(current < iterations)
            {
                Train(learningRate, currentError);
                current++;
            }
        }

        private double GetLearningRate(double error, double original)
        {
            return original * error;
        }

        public void Train(double learningRate, float error)
        {
            double currentError = Network.GetError(inputs, desiredOutputs);
            while (true)
            {
                
                currentError = Train(learningRate, currentError);
                for(int i = 0; i < inputs.Length; i++)
                {
                    Console.SetCursorPosition(0, i + 1);
                    //Console.Write("the sine of ");
                    //Console.Write(inputs[i][0]);
                    //Console.Write(" is ");
                    //Console.Write(Network.Compute(inputs[i])[0]);
                    
                    Console.Write(inputs[i][0]);
                    Console.Write(" XOR ");
                    Console.Write(inputs[i][1]);
                    Console.Write(" is ");
                    Console.Write(Network.Compute(inputs[i])[0]);
                    Console.Write($" Error: {Network.GetError(inputs[i], desiredOutputs[i])}");
                }
                ;
            }
        }
    }
}
