"use client";

import React from "react";
import {
	Search,
	Home,
	Mail,
	MessageCircle,
	User,
	Cloud,
	Settings,
	MoreVertical,
} from "lucide-react";
import { LineChart, Line, XAxis, ResponsiveContainer, Tooltip } from "recharts";
import Image from "next/image";
import SpearForm from "@/app/SpearForm/SpearForm";

const data = [
	{ name: "May", value: 20000 },
	{ name: "Jun", value: 25000 },
	{ name: "Jul", value: 22000 },
	{ name: "Aug", value: 28000 },
	{ name: "Sep", value: 24000 },
	{ name: "Oct", value: 25000 },
	{ name: "Nov", value: 27000 },
];

export default function Dashboard() {
	return (
		<div className="min-h-screen bg-[#F8F9FE] p-6">
			{/* Header */}
			<div className="flex items-center justify-between mb-8">
				<div className="flex items-center gap-4">
					<h1 className="text-xl font-semibold">
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
							className="lucide lucide-circle-slash-2"
						>
							<circle cx="12" cy="12" r="10" />
							<path d="M22 2 2 22" />
						</svg>
						SPEAR
					</h1>
					<div className="relative">
						<Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
						<input
							type="text"
							placeholder="Search"
							className="pl-10 pr-4 py-2 bg-white rounded-full w-64 text-sm focus:outline-none border border-gray-100"
						/>
					</div>
				</div>
				<div className="flex items-center gap-4">
					<div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center text-white text-sm">
						4
					</div>
					<Image
						src="https://v0.dev/placeholder-user.jpg"
						alt="Profile"
						className="rounded-full"
						width={32}
						height={32}
					/>
				</div>
			</div>

			<div className="grid grid-cols-[80px_1fr_400px] gap-6">
				{/* Sidebar */}
				<div className="space-y-6">
					<nav className="space-y-4">
						<SidebarItem
							icon={<Home className="h-5 w-5" />}
							active
						/>
						<SidebarItem icon={<Mail className="h-5 w-5" />} />
						<SidebarItem
							icon={<MessageCircle className="h-5 w-5" />}
						/>
						<SidebarItem icon={<User className="h-5 w-5" />} />
						<SidebarItem icon={<Cloud className="h-5 w-5" />} />
						<SidebarItem icon={<Settings className="h-5 w-5" />} />
					</nav>
				</div>

				{/* Main Content */}
				<div className="space-y-6">
					<h2 className="text-xl font-semibold">Dashboard</h2>

				
						<SpearForm />

					{/* Recent Transactions */}
					<div>
						<div className="flex justify-between items-center mb-4">
							<h3 className="text-lg font-medium">
								Recent Transaction
							</h3>
							<div className="flex items-center gap-2">
								<span className="text-sm text-gray-500">
									Sort by:
								</span>
								<select className="text-sm font-medium bg-transparent border-none focus:outline-none">
									<option>Recent</option>
								</select>
							</div>
						</div>
						<div className="space-y-3">
							<TransactionItem
								icon="🛍️"
								title="Shopping"
								date="05 Jan 2020 12:00"
								amount="$300"
							/>
							<TransactionItem
								icon="🛒"
								title="Grocery"
								date="12 Jan 2020 12:00"
								amount="$45"
							/>
							<TransactionItem
								icon="💪"
								title="Gym"
								date="23 Jan 2020 12:00"
								amount="$125"
							/>
							<TransactionItem
								icon="🧺"
								title="Laundry"
								date="27 Jan 2020 12:00"
								amount="$90"
							/>
							<TransactionItem
								icon="🚗"
								title="Car Repair"
								date="28 Jan 2020 12:00"
								amount="$250"
							/>
						</div>
					</div>
				</div>

				{/* Stats Section */}
				<div className="bg-white rounded-2xl p-6">
					<div className="flex justify-between items-center mb-6">
						<div>
							<div className="text-sm text-gray-500 mb-1">
								Spent This Month
							</div>
							<div className="text-2xl font-semibold">
								$25,999.00
							</div>
						</div>
						<button className="p-2 hover:bg-gray-50 rounded-full">
							<MoreVertical className="h-5 w-5 text-gray-400" />
						</button>
					</div>

					<div className="flex gap-4 mb-6">
						<button className="text-sm text-gray-500">Day</button>
						<button className="text-sm text-gray-500">Week</button>
						<button className="text-sm font-medium text-black">
							Month
						</button>
						<button className="text-sm text-gray-500">Year</button>
					</div>

					<div className="h-[200px] mb-6">
						<ResponsiveContainer width="100%" height="100%">
							<LineChart data={data}>
								<XAxis
									dataKey="name"
									axisLine={false}
									tickLine={false}
									tick={{ fontSize: 12, fill: "#9CA3AF" }}
								/>
								<Tooltip />
								<Line
									type="monotone"
									dataKey="value"
									stroke="#000"
									strokeWidth={2}
									dot={false}
								/>
							</LineChart>
						</ResponsiveContainer>
					</div>

					<div className="bg-gray-900 text-white rounded-2xl p-4">
						<div className="flex justify-between items-center mb-2">
							<div className="text-sm">Plan for 2020</div>
							<div className="text-sm">Completed</div>
						</div>
						<div className="flex items-center gap-4">
							<div className="flex-1">
								<div className="h-2 bg-gray-700 rounded-full overflow-hidden">
									<div
										className="h-full bg-white rounded-full"
										style={{ width: "75%" }}
									/>
								</div>
							</div>
							<div className="text-lg font-semibold">75%</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
}

function SidebarItem({
	icon,
	active = false,
}: {
	icon: React.ReactNode;
	active?: boolean;
}) {
	return (
		<button
			className={`w-full flex items-center justify-center p-3 rounded-xl transition-colors
        ${active ? "bg-white shadow-sm" : "hover:bg-white"}`}
		>
			{icon}
		</button>
	);
}

function TransactionItem({
	icon,
	title,
	date,
	amount,
}: {
	icon: string;
	title: string;
	date: string;
	amount: string;
}) {
	return (
		<div className="flex items-center justify-between p-4 bg-white rounded-xl shadow-sm">
			<div className="flex items-center gap-4">
				<div className="text-2xl">{icon}</div>
				<div>
					<div className="font-medium">{title}</div>
					<div className="text-sm text-gray-500">{date}</div>
				</div>
			</div>
			<div className="flex items-center gap-4">
				<div className="font-medium">{amount}</div>
				<button className="p-1 hover:bg-gray-50 rounded-full">
					<MoreVertical className="h-4 w-4 text-gray-400" />
				</button>
			</div>
		</div>
	);
}
