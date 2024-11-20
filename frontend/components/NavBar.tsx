import React from "react";
import { Button } from "@/components/ui/button";
import Link from "next/link";

const Navbar = () => {
	return (
		<nav className="flex text-center items-center justify-between px-6 py-4 border-b">
			<div className="flex items-center">
				<span className="text-2xl font-bold flex gap-2">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						width="24"
						height="24"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						strokeWidth="2"
						strokeLinecap="round"
						strokeLinejoin="round"
						className="lucide lucide-brain"
					>
						<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z" />
						<path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z" />
						<path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4" />
						<path d="M17.599 6.5a3 3 0 0 0 .399-1.375" />
						<path d="M6.003 5.125A3 3 0 0 0 6.401 6.5" />
						<path d="M3.477 10.896a4 4 0 0 1 .585-.396" />
						<path d="M19.938 10.5a4 4 0 0 1 .585.396" />
						<path d="M6 18a4 4 0 0 1-1.967-.516" />
						<path d="M19.967 17.484A4 4 0 0 1 18 18" />
					</svg>
					Cntxt Extraction
				</span>
			</div>

			<Button
				variant="default"
				className="bg-black hover:bg-gray-800 text-white p-6"
			>
        <MountainIcon />
        <Link
         href={'https://drive.google.com/drive/folders/18gujhsSBBZ2ImuoGfpKsTDYQLmQopsRE?usp=sharing'}
         target="blank"
         className="flex text-2xl"
         >
				Resrcs
        </Link>
			</Button>
		</nav>
	);
};

export default Navbar;

function MountainIcon() {
	return (
		<svg
			xmlns="http://www.w3.org/2000/svg"
			width="24"
			height="24"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="m8 3 4 8 5-5 5 15H2L8 3z" />
		</svg>
	);
}
