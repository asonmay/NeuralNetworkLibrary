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
        private ErrorFunction error;
        public double Delta;

        public Neuron(ActivationFunction activation, Neuron[] previousNerons, ErrorFunction error) 
        {
            Activation = activation;
            Dendrites = new Dendrite[previousNerons.Length];
            for(int i = 0; i < previousNerons.Length; i++)
            {
                Dendrites[i] = new Dendrite(previousNerons[i], this, 1);
            }
            this.error = error;
        }

        public void ApplyChanges()
        {
            Bias = UpdateBias;
            UpdateBias = 0;
            for(int i = 0; i < Dendrites.Length; i++)
            {
                Dendrites[i].ApplyChanges();
            }
        }

        public void BackProp(double learningRate)
        {
            double output = Compute();
            double derivitive = Delta * Activation.Derivative(output);
            for (int i = 0; i < Dendrites.Length; i++)
            {             
                Dendrites[i].UpdateWeight = Dendrites[i].Weight + (learningRate * -(derivitive * Dendrites[i].Previous.Output));
                Dendrites[i].Previous.Delta += Delta * Dendrites[i].Weight;
            }
            UpdateBias = Bias + derivitive;
            Delta = 0;
        }

        public void Randomize(Random random, int min, int max) 
        {
            Bias = random.Next(min, max);
            for(int i = 0; i < Dendrites.Length; i++)
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
            return sum + Bias;
        }
    }
}
