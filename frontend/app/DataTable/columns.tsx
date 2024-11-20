"use client"

import { ColumnDef } from "@tanstack/react-table"

// This type is used to define the shape of our data.
// You can use a Zod schema here if you want.
type predData = {
    index: string; // The starting array keys (CATEGORY, LEVEL, etc.)
    highestClass: { class: string; percentage: number }; // Class with the highest percentage
    rest: Record<string, number>; // Remaining classes with percentages
  };

export const columns: ColumnDef<predData>[] = [
  {
    accessorKey: "index",
    header: "Index",
  },
  {
    accessorKey: "highestprobclass",
    header: "Highest prob class",
  },
  {
    accessorKey: "restClasses",
    header: "Rest of the classes",
  },
]
