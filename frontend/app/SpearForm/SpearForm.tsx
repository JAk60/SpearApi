"use client";

import { Button } from "@/components/ui/button";
import { Card, CardHeader } from "@/components/ui/card";
import {
	Form,
	FormControl,
	FormDescription,
	FormField,
	FormItem,
	FormLabel,
	FormMessage,
} from "@/components/ui/form";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import axios, { AxiosResponse } from "axios";
import { Loader2 } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import * as z from "zod";
import DataTable, { DataEntry } from "../DataTable/page";

const formSchema = z.object({
	scenario: z.string().min(1, "Scenario is required"),
	model: z.string().min(1, "Model is required"),
	version: z.string().min(1, "Version is required"),
});

export default function SpearForm() {
	const [prediction, setPrediction] = useState<DataEntry[] | undefined>();
	const { toast } = useToast();
	const form = useForm<z.infer<typeof formSchema>>({
		resolver: zodResolver(formSchema),
		defaultValues: {
			scenario: "",
			model: "JL",
			version: "2",
		},
	});

	const axiosInstance = axios.create({
		baseURL: "http://127.0.0.1:8000",
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
	});

	const mutation = useMutation<
		AxiosResponse<DataEntry[]>,
		Error,
		z.infer<typeof formSchema>
	>({
		mutationFn: (values) => axiosInstance.post("/predict", values),
		onSuccess: (response) => {
			// Safely handle the response data
			const resultData = Array.isArray(response.data)
				? response.data
				: [];

			setPrediction(resultData);

			toast({
				title: "Prediction Successful",
				description: (
					<pre className="h-full mt-2 w-[340px] rounded-md bg-slate-950 p-4">
						<code className="text-white">
							<ScrollArea className="h-[200px] w-[350px] rounded-md border p-4">
								{JSON.stringify(resultData, null, 2)}
							</ScrollArea>
						</code>
					</pre>
				),
			});
		},
		onError: (error) => {
			const errorMessage = axios.isAxiosError(error)
				? error.code === "ERR_NETWORK"
					? "Unable to connect to the server. Please check if the server is running and try again."
					: error.response
					? `Server error: ${error.response.status} ${error.response.statusText}`
					: "There was a problem with your request."
				: "There was a problem with your request.";

			toast({
				variant: "destructive",
				title: "Uh oh! Something went wrong.",
				description: errorMessage,
			});
			console.error("Mutation error", error);
		},
	});

	function onSubmit(values: z.infer<typeof formSchema>) {
		mutation.mutate(values);
	}

	return (
		<>
			<Card className="p-9">
				<CardHeader className="text-2xl">
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
				</CardHeader>
				<Form {...form}>
					<form
						onSubmit={form.handleSubmit(onSubmit)}
						className="space-y-6 max-w-3xl mx-[300px] py-10"
					>
						<FormField
							control={form.control}
							name="scenario"
							render={({ field }) => (
								<FormItem>
									<FormLabel>Scenario</FormLabel>
									<FormControl>
										<Textarea
											placeholder="Please Enter the Mission Scenario"
											className="resize-none"
											{...field}
										/>
									</FormControl>
									<FormDescription>
										Enter your Scenario here.
									</FormDescription>
									<FormMessage />
								</FormItem>
							)}
						/>

						<div className="grid grid-cols-12 gap-4">
							<div className="col-span-6">
								<FormField
									control={form.control}
									name="model"
									render={({ field }) => (
										<FormItem>
											<FormLabel>Model</FormLabel>
											<Select
												onValueChange={field.onChange}
												value={field.value}
											>
												<FormControl>
													<SelectTrigger>
														<SelectValue placeholder="Select a model" />
													</SelectTrigger>
												</FormControl>
												<SelectContent>
													<SelectItem value="Bert">
														Bert
													</SelectItem>
													<SelectItem value="CAGE">
														CAGE
													</SelectItem>
													<SelectItem value="JL">
														JL
													</SelectItem>
												</SelectContent>
											</Select>
											<FormDescription>
												Select which model to use
											</FormDescription>
											<FormMessage />
										</FormItem>
									)}
								/>
							</div>

							<div className="col-span-6">
								<FormField
									control={form.control}
									name="version"
									render={({ field }) => (
										<FormItem>
											<FormLabel>Variant</FormLabel>
											<Select
												onValueChange={field.onChange}
												value={field.value}
											>
												<FormControl>
													<SelectTrigger>
														<SelectValue placeholder="Select a variant" />
													</SelectTrigger>
												</FormControl>
												<SelectContent>
													<SelectItem value="1">
														1
													</SelectItem>
													<SelectItem value="2">
														2
													</SelectItem>
													<SelectItem value="3">
														3
													</SelectItem>
												</SelectContent>
											</Select>
											<FormDescription>
												Choose a variant
											</FormDescription>
											<FormMessage />
										</FormItem>
									)}
								/>
							</div>
						</div>
						<Button
							type="submit"
							disabled={mutation.status === "pending"}
						>
							{mutation.status === "pending" ? (
								<>
									<Loader2 className="mr-2 h-4 w-4 animate-spin" />
									Submitting...
								</>
							) : (
								"Submit"
							)}
						</Button>
					</form>
				</Form>
			</Card>
			{prediction && <DataTable data={prediction} />}
		</>
	);
}
