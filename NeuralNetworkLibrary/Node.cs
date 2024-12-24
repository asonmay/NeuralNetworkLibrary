using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class Node<T> where T : IGameState<T>
    {
        public T GameState { get; set; }
        public bool IsExpanded { get; set; }
        public double W { get; set; }
        public double N { get; set; }
        public Node<T> Parent { get; set; }
        public Node<T>[] Children { get; set; }

        public Node(T gameState)
        {
            GameState = gameState;
        }

        public Node(T gameState, double w, double n, Node<T> parent)
        {
            GameState = gameState;
            W = w;
            N = n;
            Parent = parent;
            IsExpanded = false;
        }

        public double UCT(double c)
        {
            return W / N + c * Math.Sqrt(Math.Log(Parent.N)/N);
        }

        public void GenerateChildren()
        {
            T[] gameStates = GameState.GetChildren();
            Children = new Node<T>[gameStates.Length];
            for(int i = 0; i < gameStates.Length; i++)
            {
                Children[i] = new Node<T>(GameState.GetChildren()[i], 0, 0, this);
            }
        }
    }
}
