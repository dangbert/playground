import { useRouter } from 'next/router';

import Head from 'next/head'; // lets you change the head / meta tags of the doc
import Image from 'next/image';

import { UrlParams, CarEntry, baseUrl } from './index';


interface CarParams {
  imageUrl: string;
  color: string;
};

const Car: React.FC<CarParams> = ({color, imageUrl }) => {
  const router = useRouter();
  const { id } = router.query;

  return (
    <>
      <Head>
        <title>{color} {id}</title>
      </Head>
      <h1>Car: {id}</h1>

      {/* https://nextjs.org/docs/messages/no-img-element
          https://github.com/vercel/next.js/discussions/18848
      */}
      <img src={imageUrl} width="300px" />
      {/* <image src={imageUrl} width="300px" /> */}
    </>
  );
};


/**
 * tells next.js to prerender page.
 * when building the site, next.js calls this function and sends
 * the results as props to the <Car /> component.
 * 
 * this could be renamed to getServerSideProps and work work just the same,
 *   but with server side (dynamic) rendering.
 */
export async function getStaticProps( { params }: UrlParams) {
//export async function getServerSideProps( { params }: UrlParams) {
  // read data from json file in public/cars/
  const req = await fetch(`${baseUrl}/cars.json`);
  let data = await req.json();
  data = data.filter((car: CarEntry) => car.id === params.id);

  return {
    props: data[0],
  };
}

/**
 * next.js needs to know which IDs exist, so it can pre-render their pages.
 */
export async function getStaticPaths( { params }: UrlParams) {
  const req = await fetch(`${baseUrl}/cars.json`);
  const data = await req.json();

  const paths = data.map((car: any) => {
    return ({
      params: { id: car.id },
  })});

  return {
    paths,
    fallback: false, // fallback behaviour
  };
}

export default Car;