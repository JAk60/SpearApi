// import Image from "next/image";
// import SpearForm from "./SpearForm/SpearForm";
// import Navbar from "@/components/NavBar";
import Dashboard from "@/components/Dashboard";

export default function Home() {
  return (
    <>
    <Dashboard />
      {/* <Navbar /> */}
      {/* <div className="grid grid-rows-[1fr_auto] min-h-[calc(100vh-75px)] p-4 sm:p-8 font-[family-name:var(--font-geist-sans)]">
        <main className="flex flex-col items-center justify-center w-full pt-8">
          <SpearForm/>
        </main>
        <footer className="flex gap-6 flex-wrap items-center justify-center py-4"> 
          <a
            className="flex items-center gap-2 hover:underline hover:underline-offset-4"
            href="https://arxiv.org/pdf/1911.09860"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              aria-hidden
              src="/file.svg"
              alt="file icon"
              width={16}
              height={16}
            />
            Go to cage →
          </a>
          <a
            className="flex items-center gap-2 hover:underline hover:underline-offset-4"
            href="https://github.com/decile-team/spear"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              aria-hidden
              src="/globe.svg"
              alt="Globe icon"
              width={16}
              height={16}
            />
            Go to decile-spear →
          </a>
        </footer>
      </div> */}
    </>
  );
}