export const baseUrl = "http://localhost:3000"

export interface UrlParams {
  params: { id: string};
};


// entry for a car in cars.json
export interface CarEntry {
  imageUrl: string;
  color: string;
  id: string;
};

interface CarListParams {
  cars: CarEntry[];
};

// TODO: learn proper routing for link below
const CarsList: React.FC<CarListParams> = ({ cars }) => {
    return (
      <>
        <h1>Cars List</h1>

        <ol>
          {cars.map(car => (
            <li key={car.id}>
              <a href={`/cars/${car.id}`}>{car.color} {car.id}</a>
            </li>
          ))}
        </ol>
      </>
    );
};

/**
 * this function tells next.js we want this component to be rendered dynamically by the server
 * (each time someone visits a URL that renders this component).
 * 
 * Alternatively we could define getStaticProps and getStaticPaths for static rendering (at build time).
 */
export async function getServerSideProps() {
  const req = await fetch(`${baseUrl}/cars.json`);
  let data = await req.json();

  return {
    props: {
      cars: data,
    }
  };
}

export default CarsList;