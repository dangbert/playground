import { useState } from 'react';
import { DropTargetMonitor, useDrop } from 'react-dnd';
import Picture, { PictureProps } from './Picture';

export enum MyDragTypes {
  Image,
}

export interface IDragItem {
  id: number;
}

// objects that can be draggable
const pictureList: PictureProps[] = [
  {
    id: 1,
    url: 'https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Nnx8cGVvcGxlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=500&q=60',
  },
  {
    id: 2,
    url: 'https://images.unsplash.com/photo-1543269865-cbf427effbad?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTR8fHBlb3BsZXxlbnwwfHwwfHw%3D&auto=format&fit=crop&w=500&q=60',
  },
  {
    id: 3,
    url: 'https://images.unsplash.com/photo-1434725039720-aaad6dd32dfe?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NHx8bGFuZHNjYXBlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=500&q=60',
  },
  {
    id: 4,
    url: 'https://images.unsplash.com/photo-1470770841072-f978cf4d019e?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8N3x8bGFuZHNjYXBlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=500&q=60',
  },
  {
    id: 5,
    url: 'https://images.unsplash.com/photo-1433838552652-f9a46b332c40?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTF8fGxhbmRzY2FwZXxlbnwwfHwwfHw%3D&auto=format&fit=crop&w=500&q=60',
  },
];

interface DragDropProps {}
const DragDrop: React.FC<DragDropProps> = ({}) => {
  // track which pictures have been added to the board (via draggging)

  // list of pictures in the "board" section of the page
  const [board, setBoard] = useState<PictureProps[]>([]);

  const [{ isOver }, drop] = useDrop(() => ({
    // define which types of objects we want to accept
    accept: String(MyDragTypes.Image),
    drop: (item: IDragItem) => addImageToBoard(item.id),
    collect: (monitor: DropTargetMonitor) => ({
      isOver: !!monitor.isOver(),
    }),
  }));

  const addImageToBoard = (id: number) => {
    const found = pictureList.find((item) => item.id === id);
    if (!found) {
      console.error(`unable to find image ${id} to add to board`);
      return;
    }
    setBoard((prev) => [...prev, found]);
  };

  // list of pictures not (yet) in the "board"
  const unusedPictures = pictureList.filter(
    (pic) => !board.find((item) => item.id === pic.id)
  );

  return (
    <>
      <div className="pictures">
        {unusedPictures.map((pic) => (
          <Picture {...pic} key={pic.id} />
        ))}
      </div>
      <div
        className="board"
        ref={drop}
        style={
          isOver ? { border: '8px solid gold' } : { border: '8px solid black' }
        }
      >
        {board.map((pic) => (
          <Picture {...pic} key={pic.id} />
        ))}
      </div>
    </>
  );
};

export default DragDrop;
