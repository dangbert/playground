import React from 'react';
import logo from './logo.svg';
import './App.css';
import DragDrop from './components/DragDrop';

import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

function App() {
  return (
    <div className="page">
      <DndProvider backend={HTML5Backend}>
        <DragDrop />
      </DndProvider>
    </div>
  );
}

export default App;
