export default function Sidebar({ history }) {
  return (
    <aside className="sidebar">
      <h2>🕘 History</h2>
      <ul>
        {history.map((h, i) => (
          <li key={i}>{h.input.slice(0, 40)}...</li>
        ))}
      </ul>
    </aside>
  );
}
