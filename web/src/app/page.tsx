import styles from "./page.module.css";


export default function Home() {
  return (
    <>
      <main className={styles.main}>
        <section className={styles.hero}>
          <h1 className={`${styles.title} glow-text`}>
            Zero-Modification <br />
            <span className="gradient-text">Semantic Steganography</span>
          </h1>
          <p className={styles.subtitle}>
            Hiding meaning in plain sight without altering a single bit. An AI-driven approach to covert communication.
          </p>
        </section>
      </main>
    </>
  );
}
