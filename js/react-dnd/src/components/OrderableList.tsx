// OrderableList.tsx
import React, { useState } from 'react';
import {
  DndProvider,
  useDrag,
  useDrop,
  DragPreviewImage,
  DragLayerMonitor,
} from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

const ItemType = 'ORDERABLE_ITEM';

const OrderableList: React.FC = () => {
  const [items, setItems] = useState([
    { id: 1, text: 'Item 1' },
    { id: 2, text: 'Item 2' },
    { id: 3, text: 'Item 3' },
    { id: 4, text: 'Item 4' },
  ]);

  const moveItem = (fromIndex: number, toIndex: number) => {
    const updatedItems = [...items];
    const [movedItem] = updatedItems.splice(fromIndex, 1);
    updatedItems.splice(toIndex, 0, movedItem);
    setItems(updatedItems);
  };

  const getPreviewImage = (monitor: DragLayerMonitor, props: any) => {
    const item = props.item;
    if (!item) return null;

    const previewNode = document.createElement('div');
    previewNode.style.border = '2px solid #000';
    previewNode.style.backgroundColor = 'rgba(0, 0, 0, 0.1)';
    previewNode.style.padding = '8px';
    previewNode.innerText = item.text;

    return previewNode;
  };

  return (
    <DndProvider backend={HTML5Backend}>
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {items.map((item, index) => (
          <OrderableItem
            key={item.id}
            id={item.id}
            text={item.text}
            index={index}
            moveItem={moveItem}
          />
        ))}
      </div>
      {/* @ts-ignore */}
      <DragPreviewImage connect={getPreviewImage} />
    </DndProvider>
  );
};

interface OrderableItemProps {
  id: number;
  text: string;
  index: number;
  moveItem: (fromIndex: number, toIndex: number) => void;
}

const OrderableItem: React.FC<OrderableItemProps> = ({
  id,
  text,
  index,
  moveItem,
}) => {
  const [collected, dragRef, preview] = useDrag({
    type: ItemType,
    item: { id, index },
  });

  // @ts-ignore
  const [{ isOver }, dropRef] = useDrop({
    accept: ItemType,
    hover: (item: { id: number; index: number }) => {
      if (item.index !== index) {
        moveItem(item.index, index);
        item.index = index;
      }
    },
  });

  return (
    <>
      <DragPreviewImage connect={preview} src="" />
      <div
        ref={(node) => dragRef(dropRef(node))}
        style={{
          borderBottom: isOver ? '2px solid #ccc' : 'inherit',
          padding: '8px',
          margin: '4px',
          backgroundColor: 'white',
        }}
      >
        {text}
      </div>
    </>
  );
};

export default OrderableList;
