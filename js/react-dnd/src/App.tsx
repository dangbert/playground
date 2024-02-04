import React from 'react';
import './App.css';
import DragDrop from './components/DragDrop';
import OrderableList from './components/OrderableList';

import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

function App() {
  return (
    <div className="page">
      <DndProvider backend={HTML5Backend}>
        {/* <DragDrop /> */}
        <OrderableList />
      </DndProvider>
    </div>
  );
}

export default App;
