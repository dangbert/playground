import Basics from './learning/Basics';
import Orientation from './learning/Orientation';
import WelcomeScreen from './screens/WelcomeScreen';
import ViewImageScreeen from './screens/ViewImageScreen';
import { useState } from 'react';

export default function App() {
  const [curScreen, setCurScreen] = useState('welcome');
  // return <Basics />;
  // return <Orientation />;
  if (curScreen === 'welcome') {
    return (
      <WelcomeScreen changeScreen={(next: string) => setCurScreen(next)} />
    );
  } else if (curScreen === 'image') {
    return (
      <ViewImageScreeen changeScreen={(next: string) => setCurScreen(next)} />
    );
  }
}
