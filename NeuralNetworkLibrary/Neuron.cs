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
        public double Bias;
        public double UpdateBias;
        public Dendrite[] Dendrites;
        public double Output { get; set; }
        public double Input { get; private set; }
        public ActivationFunction Activation { get; set; }
        public double Delta;

        private double previousUpdaetBias;
        public Neuron(ActivationFunction activation, Neuron[] previousNerons) 
        {
            Activation = activation;
            Dendrites = new Dendrite[previousNerons.Length];
            for(int i = 0; i < previousNerons.Length; i++)
            {
                Dendrites[i] = new Dendrite(previousNerons[i], this, 1);
            }
        }

        public void ApplyChanges(double momentum)
        {
            UpdateBias += previousUpdaetBias * momentum;
            Bias += UpdateBias;
            previousUpdaetBias = UpdateBias;
            UpdateBias = 0;
            for(int i = 0; i < Dendrites.Length; i++)
            {
                Dendrites[i].ApplyChanges(momentum);
            }
        }

        public void BackProp(double learningRate)
        {
            double output = Compute();
            double derivative = Delta * Activation.Derivative(output);
            for (int i = 0; i < Dendrites.Length; i++)
            {             
                Dendrites[i].UpdateWeight += learningRate * -(derivative * Dendrites[i].Previous.Output);

                Dendrites[i].Previous.Delta = Delta * Dendrites[i].Weight * Activation.Derivative(output);
            }
            UpdateBias += learningRate * -derivative;
            Delta = 0;
        }

        public void Randomize(Random random, double min, double max) 
        {
            Bias = random.NextDouble() * (Math.Abs(min) + Math.Abs(max) + 1) + min;
            for (int i = 0; i < Dendrites.Length; i++)
            {
                Dendrites[i].Weight = random.NextDouble() * (Math.Abs(min) + Math.Abs(max) + 1) + min;
            }
        }

        public double Compute() 
        {
            double sum = 0;
            for(int i = 0; i < Dendrites.Length; i++)
            {
                sum += Dendrites[i].Compute();
            }
            return Activation.Function(sum + Bias); 
        }
    }
}
