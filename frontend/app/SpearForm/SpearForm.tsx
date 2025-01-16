import { useState, useEffect, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";

// Type definitions
interface QuestionData {
  question: string;
  metadata: {
    layer_type: string;
    layer_key: string;
    class_1: string;
    class_2: string;
    prob_diff: number;
  };
}

interface CompletionData {
  status: 'completed';
  refined_classifications: Record<string, Record<string, number>>;
  high_confidence_classifications: Record<string, Record<string, number>>;
  timing_summary: {
    total_time: number;
    connection_time: number;
    checkpoints: Record<string, number>;
  };
  log_file?: string;
}

interface ErrorData {
  error: string;
}

type WSMessage = QuestionData | CompletionData | ErrorData;

// Type guard functions
function isCompletionData(data: WSMessage): data is CompletionData {
  return (
    (data as CompletionData).status === 'completed' &&
    typeof (data as CompletionData).refined_classifications === 'object' &&
    (data as CompletionData).refined_classifications !== null &&
    (data as CompletionData).high_confidence_classifications !== null &&
    typeof (data as CompletionData).timing_summary === 'object' &&
    (data as CompletionData).timing_summary !== null
  );
}



function isQuestionData(data: WSMessage): data is QuestionData {
  return typeof (data as QuestionData).question === 'string' && 
         (data as QuestionData).metadata !== undefined &&
         typeof (data as QuestionData).metadata.layer_type === 'string';
}


const formSchema = z.object({
  scenario: z.string().min(1, "Scenario is required"),
});

type FormValues = z.infer<typeof formSchema>;

export default function SpearForm() {
  const [currentQuestion, setCurrentQuestion] = useState<QuestionData | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [questionHistory, setQuestionHistory] = useState<Array<{question: string, answer: string}>>([]);
  const [refinedClassifications, setRefinedClassifications] = useState<Record<string, Record<string, number>> | null>(null);
  const [highClassifications, setHighClassifications] = useState<Record<string, Record<string, number>> | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const { toast } = useToast();

  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      scenario: "",
    },
  });

  const connectWebSocket = (scenario: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }

    const url = new URL('ws://127.0.0.1:8000/predict');
    url.searchParams.append('scenario', scenario);
    url.searchParams.append('version', '6');
    url.searchParams.append('model', 'BERT_classifier');

    wsRef.current = new WebSocket(url.toString());
    
    wsRef.current.onopen = () => {
      console.log('WebSocket connection established');
      setIsProcessing(true);
      setQuestionHistory([]);
      setRefinedClassifications(null);
      setHighClassifications(null);
    };

    wsRef.current.onmessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data) as WSMessage;
      
      debugger
      if (isCompletionData(data)) {
        setIsProcessing(false);
        setCurrentQuestion(null);
        setRefinedClassifications(data.refined_classifications);
        setHighClassifications(data.high_confidence_classifications)
        console.log('Refined Classifications:', data.refined_classifications);
        console.log('high_confidence_classifications:', data.high_confidence_classifications);
        toast({
          title: "Processing Complete",
          description: `Analysis completed in ${data.timing_summary.total_time.toFixed(2)} seconds`,
        });
        wsRef.current?.close();
      } else if (isQuestionData(data)) {
        setCurrentQuestion(data);
      } else {
        // Handle error case
        toast({
          variant: "destructive",
          title: "Error",
          description: 'error' in data ? data.error : "Unknown error occurred",
        });
        setIsProcessing(false);
      }
    };

    wsRef.current.onerror = () => {
      toast({
        variant: "destructive",
        title: "WebSocket Error",
        description: "Connection failed or encountered an error",
      });
      setIsProcessing(false);
    };

    wsRef.current.onclose = () => {
      setIsProcessing(false);
    };
  };

  const handleAnswer = (answer: 'yes' | 'no') => {
    if (wsRef.current?.readyState === WebSocket.OPEN && currentQuestion) {
      wsRef.current.send(answer);
      setQuestionHistory(prev => [...prev, {
        question: currentQuestion.question,
        answer: answer
      }]);
      setCurrentQuestion(null);
    }
  };

  const onSubmit = (values: FormValues) => {
    setIsProcessing(true);
    connectWebSocket(values.scenario);
  };

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  return (
    <div className="space-y-6 mx-auto py-10">
      <Card className="p-9">
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
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
                      disabled={isProcessing}
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <Button type="submit" disabled={isProcessing}>
              {isProcessing ? "Processing..." : "Submit"}
            </Button>
          </form>
        </Form>
      </Card>

      {currentQuestion && (
        <Card className="p-6">
          <div className="space-y-4">
            <h3 className="text-lg font-medium">Question:</h3>
            <p className="text-gray-700">{currentQuestion.question}</p>
            <div className="flex space-x-4">
              <Button 
                onClick={() => handleAnswer("yes")}
                className="bg-green-600 hover:bg-green-700"
              >
                Yes
              </Button>
              <Button 
                onClick={() => handleAnswer("no")}
                className="bg-red-600 hover:bg-red-700"
              >
                No
              </Button>
            </div>
          </div>
        </Card>
      )}

      {questionHistory.length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-medium mb-4">Question History</h3>
          <div className="space-y-4">
            {questionHistory.map((item, index) => (
              <div key={index} className="border-b pb-2">
                <p className="font-medium">Q: {item.question}</p>
                <p className="text-gray-600">A: {item.answer}</p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {refinedClassifications && (
        <Card className="p-6">
          <h3 className="text-lg font-medium mb-4">Final Classifications</h3>
          <div className="space-y-4">
            {Object.entries(refinedClassifications).map(([layer, classifications]) => (
              <div key={layer} className="border-b pb-4">
                <h4 className="font-medium mb-2">{layer}</h4>
                <div className="pl-4">
                  {Object.entries(classifications).map(([className, probability]) => (
                    <div key={className} className="text-sm">
                      <span className="font-medium">{className}:</span>{' '}
                      <span className="text-gray-600">{(probability * 100).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
      {highClassifications && (
        <Card className="p-6">
          <h3 className="text-lg font-medium mb-4">Final Classifications</h3>
          <div className="space-y-4">
            {Object.entries(highClassifications).map(([layer, classifications]) => (
              <div key={layer} className="border-b pb-4">
                <h4 className="font-medium mb-2">{layer}</h4>
                <div className="pl-4">
                  {Object.entries(classifications).map(([className, probability]) => (
                    <div key={className} className="text-sm">
                      <span className="font-medium">{className}:</span>{' '}
                      <span className="text-gray-600">{(probability * 100).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}