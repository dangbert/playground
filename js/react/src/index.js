// created based on tutorial https://reactjs.org/tutorial/tutorial.html
import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';

class Game extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            history: [
                {squares: Array(9).fill(null)}
            ],
            xIsNext: true,
            atMove: 0 // which move of the game we're currently viewing
        }
    }
    handleClick(i) {
        var history = this.state.history;
        const squares = history[this.state.atMove].squares.slice(); // returns a shallow copy?

        if (calculateWinner(squares) || squares[i]) {
            return;
        }
        squares[i] = this.state.xIsNext ? 'X' : 'O';
        
        history = history.slice(0, this.state.atMove+1);
        history.push({"squares": squares})
        // in React you can't change this.state directly:
        this.setState({
            history: history,
            xIsNext: !this.state.xIsNext,
            atMove: this.state.atMove + 1 // which move in game we're currently at
        });
    }
    jumpTo(moveNum) {
        // only need to set the values that changed:
        this.setState({
            xIsNext: (moveNum % 2 === 0),
            atMove: moveNum
        });
    }
    render() {
        const history = this.state.history;
        const current = history[this.state.atMove];
        const winner = calculateWinner(current.squares);
        let status = 'Next player: ' + (this.state.xIsNext ? 'X' : 'O');
        if (winner) {
            status = "Winner: " + winner;
        }

        // create array of move history html
        const moves = history.map((state, index) => {
            const desc = index !== 0 ? "Go to move #" + index : "Go to game start";
            // keys are needed for list items to help react identify future changes
            //   https://reactjs.org/tutorial/tutorial.html#picking-a-key
            return (
                <li key={index}>
                    <button onClick={() => this.jumpTo(index)}>
                        {desc}
                    </button>
                </li>
            );
        });
        return (
            <div className="game">
                <div className="game-board">
                    <Board
                        onClick={(i) => this.handleClick(i)}
                        squares={current.squares}
                    />
                </div>
                <div className="game-info">
                    <div>{status}</div>
                    <ol>{moves}</ol>
                </div>
            </div>
        );
    }
}

class Board extends React.Component {
    renderSquare(i) {
        return (
            <Square
                value={this.props.squares[i]}
                onClick={() => this.props.onClick(i)}
            />
        );
    }
    render() {
        return (
            <div>
                <div className="board-row">
                    {this.renderSquare(0)}
                    {this.renderSquare(1)}
                    {this.renderSquare(2)}
                </div>
                <div className="board-row">
                    {this.renderSquare(3)}
                    {this.renderSquare(4)}
                    {this.renderSquare(5)}
                </div>
                <div className="board-row">
                    {this.renderSquare(6)}
                    {this.renderSquare(7)}
                    {this.renderSquare(8)}
                </div>
            </div>
        );
    }
}

// using a function component here for simplicity (component only needs a render function)
//   https://reactjs.org/tutorial/tutorial.html#function-components
function Square(props) {
    //<button className="square" onClick={function() { alert('clicked'); } }>
    return (
        <button className="square" onClick={() => props.onClick()}>
            {props.value}
        </button>
    );
}
// ========================================

ReactDOM.render(
    <Game />,
    document.getElementById('root')
);

/*
 * Given an array of 9 squares, this function will check for a winner and
 * return 'X', 'O', or null as appropriate.
 */
function calculateWinner(squares) {
    const lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6],
    ];
    for (let i = 0; i < lines.length; i++) {
        const [a, b, c] = lines[i];
        if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
            return squares[a];
        }
    }
    return null;
}
