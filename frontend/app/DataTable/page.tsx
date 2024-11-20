"use client"

import * as React from "react"
import {
  ColumnDef,
  ColumnFiltersState,
  SortingState,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table"
import { ArrowUpDown, ChevronDown } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

export interface DataItem {
  category: string
  value: number
}

export interface DataEntry {
  name: string
  items: DataItem[]
}

// const data: DataEntry[] = [
//   {
//     name: "CATEGORY",
//     items: [
//       { category: "Maintenance", value: 0.01562468178214094 },
//       { category: "Mission", value: 0.9843753182178591 },
//     ],
//   },
//   {
//     name: "MissionType",
//     items: [
//       { category: "Combat", value: 0.021929134051416464 },
//       { category: "Exercise", value: 0.005901261701113969 },
//       { category: "Fleet", value: 0.030591099339972765 },
//       { category: "Humanitarian", value: 0.003434351819713305 },
//       { category: "Not Applicable", value: 0.9127234609461983 },
//       { category: "Sortie", value: 0.010861962492230226 },
//       { category: "Support", value: 0.014558729649354953 },
//     ],
//   },
//   {
//     name: "LEVEL",
//     items: [
//       { category: "Equipment", value: 0.07372346930001816 },
//       { category: "Fleet", value: 0.663810532714513 },
//       { category: "Ship", value: 0.2624659979854687 },
//     ],
//   },
//   {
//     name: "Action",
//     items: [
//       { category: "Evaluate", value: 0.0014391637077884854 },
//       { category: "Select K out of N", value: 0.9985608362922115 },
//     ],
//   },
//   {
//     name: "Entity",
//     items: [
//       { category: "Equipment", value: 0.0014095238705252883 },
//       { category: "Ship", value: 0.9985778553132679 },
//       { category: "Workshop", value: 0.000012620816206867516 },
//     ],
//   },
//   {
//     name: "From",
//     items: [
//       { category: "Fleet", value: 0.9717664545702206 },
//       { category: "Ships", value: 0.028233545429779558 },
//     ],
//   },
//   {
//     name: "Time",
//     items: [
//       { category: "24 hours", value: 0.1134386409388377 },
//       { category: "48 hours", value: 0.1763621468927109 },
//       { category: "7 days", value: 0.0000583879936326557 },
//       { category: "Immediate", value: 0.7101408241748187 },
//     ],
//   },
//   {
//     name: "Location",
//     items: [
//       { category: "K", value: 0.05823728500371239 },
//       { category: "Sea", value: 0.9344431927260146 },
//       { category: "Ship", value: 0.007163513611585298 },
//       { category: "Workshop", value: 0.000024220291423194948 },
//       { category: "Yard", value: 0.0001317883672645022 },
//     ],
//   },
//   {
//     name: "Task Objective",
//     items: [
//       { category: "Interrogation and interception", value: 0.9993050442110057 },
//       { category: "Maintenance scheduling", value: 0.0006844991416043553 },
//       { category: "Missile firing", value: 0.000010456647389969938 },
//     ],
//   },
//   {
//     name: "Objective function",
//     items: [
//       { category: "Maximum conformance", value: 0.0001126616743448018 },
//       { category: "Minimum risk", value: 0.9550866131818241 },
//       { category: "Minimum time", value: 0.04480072514383124 },
//     ],
//   },
//   {
//     name: "Hard Constraints",
//     items: [
//       { category: "Balancing loads,Reliability,Risk score", value: 0.06951089769727424 },
//       { category: "Capability,Speed,Endurance,Ration,Fuel,Spares availability", value: 0.6834555833684827 },
//       { category: "Manpower availability,Spares availability", value: 0.01636523568178639 },
//       { category: "Reliability,Conformance ", value: 0.0006493331041651397 },
//       { category: "Reliability,Risk score", value: 0.11286491048046787 },
//       { category: "Spares availability,Reliability,Conformance ", value: 0.11715403966782369 },
//     ],
//   },
//   {
//     name: "Soft Constraints",
//     items: [
//       { category: "Activity sequence", value: 0.07080450863859995 },
//       { category: "Not Applicable", value: 0.0037527025095160688 },
//       { category: "Ship class", value: 0.0006978242943721681 },
//       { category: "Ship class,Fleet availability", value: 0.9195456388332924 },
//       { category: "Working hours,Manpower availability", value: 0.005199325724219221 },
//     ],
//   },
// ]

const columns: ColumnDef<DataEntry>[] = [
  {
    accessorKey: "name",
    header: "Category",
    cell: ({ row }) => <div className="font-medium">{row.getValue("name")}</div>,
  },
  {
    accessorKey: "items",
    header: ({ column }) => {
      return (
        <Button
          variant="ghost"
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        >
          Subcategories
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      )
    },
    cell: ({ row }) => {
      const items: DataItem[] = row.getValue("items")
      return (
        <div>
          {items.map((item, index) => (
            <div key={index} className="flex justify-between">
              <span>{item.category}:</span>
              <span>{(item.value * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      )
    },
  },
]
interface predictionDataProp {
    data: DataEntry[];
  }
  
export default function DataTable({data}:predictionDataProp) {
  const [sorting, setSorting] = React.useState<SortingState>([])
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    []
  )
  const [columnVisibility, setColumnVisibility] = React.useState({})

  const table = useReactTable({
    data,
    columns,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    onColumnVisibilityChange: setColumnVisibility,
    state: {
      sorting,
      columnFilters,
      columnVisibility,
    },
  })

  return (
    <div className="w-full">
      <div className="flex items-center py-4">
        <Input
          placeholder="Filter categories..."
          value={(table.getColumn("name")?.getFilterValue() as string) ?? ""}
          onChange={(event) =>
            table.getColumn("name")?.setFilterValue(event.target.value)
          }
          className="max-w-sm"
        />
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" className="ml-auto">
              Columns <ChevronDown className="ml-2 h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {table
              .getAllColumns()
              .filter((column) => column.getCanHide())
              .map((column) => {
                return (
                  <DropdownMenuCheckboxItem
                    key={column.id}
                    className="capitalize"
                    checked={column.getIsVisible()}
                    onCheckedChange={(value) =>
                      column.toggleVisibility(!!value)
                    }
                  >
                    {column.id}
                  </DropdownMenuCheckboxItem>
                )
              })}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  return (
                    <TableHead key={header.id}>
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
                    </TableHead>
                  )
                })}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  data-state={row.getIsSelected() && "selected"}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-24 text-center"
                >
                  No results.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
      <div className="flex items-center justify-end space-x-2 py-4">
        <div className="flex-1 text-sm text-muted-foreground">
          {table.getFilteredRowModel().rows.length} row(s) total
        </div>
        <div className="space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  )
}