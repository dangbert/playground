import {} from 'react-dnd';
import Picture, { PictureProps } from './Picture';

// objects that can be draggable
const PictureList: PictureProps[] = [
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
  return (
    <>
      <div className="pictures">
        {PictureList.map((pic) => (
          <Picture {...pic} key={pic.id} />
        ))}
      </div>
      <div className="board"></div>
    </>
  );
};

export default DragDrop;
