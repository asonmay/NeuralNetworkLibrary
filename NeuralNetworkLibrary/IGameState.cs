using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public interface IGameState<T> where T : IGameState<T>
    {
        public bool IsWin { get; set; }
        public bool IsTie { get; }
        public bool IsLoss { get; }
        public bool IsTerminal { get; }
        public T[] GetChildren();
    } 
}
