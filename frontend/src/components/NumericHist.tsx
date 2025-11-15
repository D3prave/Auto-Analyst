// src/components/NumericHist.tsx

import Plot from "react-plotly.js";

interface Props {
  column: string;
  values: number[];
}

const NumericHist = ({ column, values }: Props) => {
  if (!values.length) return null;

  return (
    <Plot
      data={[
        {
          x: values,
          type: "histogram",
        },
      ]}
      layout={{
        title: `Histogram of ${column}`,
        margin: { l: 40, r: 10, t: 40, b: 40 },
        autosize: true,
      }}
      style={{ width: "100%", height: "260px" }}
      config={{ displayModeBar: false, responsive: true }}
    />
  );
};

export default NumericHist;
