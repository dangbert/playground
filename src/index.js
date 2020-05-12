// created based on tutorial https://reactjs.org/tutorial/tutorial.html
import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';

class Game extends React.Component {
    render() {
        return (
            <div className="game">
                <div className="game-board">
                    <Board />
                </div>
                <div className="game-info">
                    <div>{/* status */}</div>
                    <ol>{/* TODO */}</ol>
                </div>
            </div>
        );
    }
}

class Board extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            squares: Array(9).fill(null),
            xIsNext: true
        }
    }
    handleClick(i) {
        const squares = this.state.squares.slice(); // returns a shallow copy?
        if (calculateWinner(squares) || squares[i]) {
            return;
        }
        // in React you shouldn't change this.state directly:
        squares[i] = this.state.xIsNext ? 'X' : 'O';
        this.setState({squares: squares, xIsNext: !this.state.xIsNext});
    }
    renderSquare(i) {
        return (
            <Square
                value={this.state.squares[i]}
                onClick={() => this.handleClick(i)}
            />
        );
    }
    render() {
        const winner = calculateWinner(this.state.squares);
        let status = 'Next player: ' + (this.state.xIsNext ? 'X' : 'O');
        if (winner) {
            status = "Winner: " + winner;
        }

        return (
            <div>
                <div className="status">{status}</div>
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
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ];
    for (let i = 0; i < lines.length; i++) {
        const [a, b, c] = lines[i];
        if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
            return squares[a];
        }
    }
    return null;
}
