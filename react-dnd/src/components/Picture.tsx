import { DragSourceMonitor, useDrag } from 'react-dnd';
import { MyDragTypes, IDragItem } from './DragDrop';

/**
 * Draggable picture object.
 */
export interface PictureProps {
  id: number;
  url: string;
}
const Picture: React.FC<PictureProps> = ({ id, url }) => {
  const [{ isDragging }, drag] = useDrag(() => ({
    type: String(MyDragTypes.Image),
    // properties made available to anything watching the drag/drop behaviour of this item:
    item: { id },
    // lets you define different states and props to be accessible:
    collect: (monitor: DragSourceMonitor) => ({
      isDragging: !!monitor.isDragging(),
    }),
  }));

  return (
    <div>
      <img
        ref={drag}
        src={url}
        alt={`${id}`}
        width="100%"
        style={
          isDragging
            ? {
                border: '5px solid purple',
              }
            : {}
        }
      />
    </div>
  );
};

export default Picture;
